classdef UAVNavEnv < rl.env.MATLABEnvironment
    %UAVNAVENV  Quadrotor visual-navigation environment for Reinforcement Learning Toolbox.
    %
    %   The UAV must fly from a random start to a random 3D goal through a
    %   field of static pillars and moving obstacles using
    %     * observation 1 "image": H x W x K stack of camera frames (gray or depth)
    %     * observation 2 "state": goal distance / bearing / height difference,
    %       altitude, body velocities, yaw rate, remaining time
    %       (+ CNN obstacle-detector outputs when UseFeatures = true)
    %   Actions: 8 discrete velocity / yaw-rate set-points (see uavActions).
    %
    %   env = UAVNavEnv(cfg)                                   % oracle perception, no shield
    %   env = UAVNavEnv(cfg, Detector=detNet, PerceptionSource="detector", ...
    %                   UseFeatures=true, ShieldEnabled=true, Seed=1);
    %   obs = resetWithSeed(env, 42);
    %   [obs, reward, isDone, info] = step(env, 1);
    %
    %   Privileged simulator data (true depth image, true sector distances) is
    %   only used to label training data and for "oracle" ablations.

    properties
        Cfg
        Actions
        Camera
        Planner
        World
        Detector = []
        PerceptionSource = "oracle"
        UseFeatures = false
        Shield = []
        Difficulty = 1
        CurriculumEpisodes = 0
        Rng
        % ---- UAV state
        Pos = zeros(3, 1)
        Vel = zeros(3, 1)
        Yaw = 0
        YawRate = 0
        Start = zeros(3, 1)
        Goal = zeros(3, 1)
        StepCount = 0
        PathLength = 0
        MinClearance = inf
        PrevDist = 0
        StartDist = 0
        Trail = zeros(3, 0)
        LastAction = 0
        Interventions = 0
        % ---- sensing
        Frames
        Depth
        SectorDistTrue
        Percept
        Mask
        ShieldInfo = struct()
        LastInfo = struct()
        Light = [0.4; 0.3; 0.866]
        Contrast = 1
        Brightness = 0
        ObsClass = "single"
        StateDim = 10
        NumSectors = 15
        % ---- bookkeeping
        Done = false
        Started = false
        EpisodeCount = 0
        EpisodeLog = struct("outcome", {}, "steps", {}, "pathLength", {}, ...
            "startDistance", {}, "difficulty", {}, "interventions", {})
    end

    methods
        function this = UAVNavEnv(cfg, options)
            arguments
                cfg (1,1) struct
                options.Detector = []
                options.PerceptionSource (1,1) string = "oracle"
                options.UseFeatures (1,1) logical = false
                options.ShieldEnabled (1,1) logical = false
                options.Seed (1,1) double = 0
            end
            nS = cfg.env.sectors.rows * cfg.env.sectors.cols;
            stateDim = 10 + double(options.UseFeatures) * (2 * nS + 1);
            s = cfg.env.camera.imgSize;
            k = cfg.env.camera.frameStack;
            obsInfo = [numericSpec([s s k], 0, 1), numericSpec([stateDim 1], -inf, inf)];
            obsInfo(1).Name = "image";
            obsInfo(2).Name = "state";
            actInfo = rlFiniteSetSpec(1:8);
            actInfo.Name = "action";
            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            this.Cfg = cfg;
            this.Actions = uavActions(cfg.env.speedScale);
            this.Camera = makeCamera(s, cfg.env.camera.fovDeg, cfg.env.camera.fogDistance);
            this.Planner = makeReactivePlanner(cfg, this.Actions);
            this.Detector = options.Detector;
            this.PerceptionSource = options.PerceptionSource;
            if this.PerceptionSource == "detector" && isempty(this.Detector)
                error("UAVNavEnv:noDetector", ...
                    "PerceptionSource ""detector"" needs a trained detector (run trainDetector) - or use ""oracle"".");
            end
            this.UseFeatures = options.UseFeatures;
            this.StateDim = stateDim;
            this.NumSectors = nS;
            if options.ShieldEnabled
                this.Shield = SafetyShield(cfg, this.Actions);
            end
            this.Difficulty = cfg.env.difficulty;
            this.Rng = RandStream("mt19937ar", "Seed", options.Seed);
            this.Mask = true(1, numel(this.Actions));
            try
                this.ObsClass = string(obsInfo(1).DataType);
            catch
                this.ObsClass = "double";
            end
        end

        % ================================================================ RL API
        function obs = reset(this)
            e = this.Cfg.env;
            if this.Started && ~this.Done && this.StepCount > 0
                this.logEpisode("timeout");          % previous episode hit the time limit
            end
            if this.CurriculumEpisodes > 0
                this.Difficulty = min(1, 0.1 + 0.9 * this.EpisodeCount / this.CurriculumEpisodes);
            end
            rs = this.Rng;
            L = e.arenaSize;
            margin = 3;
            this.Start = [unif(rs, [margin, L - margin]); unif(rs, [margin, L - margin]); unif(rs, e.startAltitude)];
            goal = this.Start;
            for attempt = 1:200
                goal = [unif(rs, [margin, L - margin]); unif(rs, [margin, L - margin]); unif(rs, e.goalAltitude)];
                if norm(goal(1:2) - this.Start(1:2)) >= e.minStartGoalDist
                    break
                end
            end
            this.Goal = goal;
            this.World = generateWorld(rs, e, this.countFor(e.nStatic), this.countFor(e.nDynamic), ...
                this.Start, this.Goal);

            this.Pos = this.Start;
            this.Vel = zeros(3, 1);
            this.Yaw = unif(rs, [-pi, pi]);
            this.YawRate = 0;
            this.StepCount = 0;
            this.PathLength = 0;
            this.Interventions = 0;
            this.LastAction = 0;
            this.PrevDist = norm(this.Goal - this.Pos);
            this.StartDist = this.PrevDist;
            this.Trail = this.Pos;
            this.randomizeAppearance();
            if ~isempty(this.Shield)
                this.Shield.reset();
            end
            frame = this.renderFrame();
            this.Frames = repmat(frame, 1, 1, e.camera.frameStack);
            clearance = worldClearance(this.World, this.Pos);
            this.MinClearance = clearance;
            this.Done = false;
            this.Started = true;
            this.updatePerception();
            this.LastInfo = this.makeInfo(false, false, false, clearance, "", 0, false, 0);
            obs = this.observation();
        end

        function [obs, reward, isDone, info] = step(this, action)
            if iscell(action)
                action = action{1};
            end
            if this.Done
                error("UAVNavEnv:episodeOver", "The episode has ended - call reset first.");
            end
            e = this.Cfg.env;
            requested = double(action);
            a = requested;
            shielded = false;
            if ~isempty(this.Shield) && ~this.Mask(a)
                a = this.safeSubstitute();
                shielded = true;
                this.Interventions = this.Interventions + 1;
            end
            act = this.Actions(a);

            collision = false;
            [clearance, kind] = worldClearance(this.World, this.Pos);
            for k = 1:e.actionRepeat
                this.integrate(act, e.dt);
                this.World = stepMovers(this.World, e.dt, this.Rng, e.moverTurnProb, e.moverAltitude);
                [clearance, kind] = worldClearance(this.World, this.Pos);
                this.MinClearance = min(this.MinClearance, clearance);
                if clearance < e.uavRadius
                    collision = true;
                    break
                end
            end
            this.StepCount = this.StepCount + 1;
            this.LastAction = a;
            this.Trail(:, end + 1) = this.Pos;

            dist = norm(this.Goal - this.Pos);
            reached = ~collision && dist < e.goalRadius;
            rw = e.reward;
            reward = rw.progress * (this.PrevDist - dist) + rw.step;
            if clearance < e.safeDistance
                reward = reward + rw.proximity * (e.safeDistance - max(clearance, 0)) / e.safeDistance;
            end
            if shielded
                reward = reward + rw.shieldPenalty;
            end
            if collision
                reward = reward + rw.collision;
            end
            if reached
                reward = reward + rw.goal;
            end
            this.PrevDist = dist;

            isDone = collision || reached;
            truncated = ~isDone && this.StepCount >= e.maxSteps;
            frame = this.renderFrame();
            this.Frames = cat(3, this.Frames(:, :, 2:end), frame);
            this.updatePerception();
            obstacle = "";
            if collision
                obstacle = kind;
            end
            this.LastInfo = this.makeInfo(collision, reached, truncated, clearance, obstacle, requested, shielded, a);
            info = this.LastInfo;
            if isDone
                this.Done = true;
                if reached
                    this.logEpisode("goal");
                else
                    this.logEpisode("collision");
                end
            end
            obs = this.observation();
        end

        % ============================================================ helpers
        function obs = resetWithSeed(this, seed)
            %RESETWITHSEED  Reset with a reproducible scenario.
            this.Rng = RandStream("mt19937ar", "Seed", seed);
            obs = this.reset();
        end

        function setDifficulty(this, d)
            this.Difficulty = min(max(d, 0), 1);
        end

        function n = countFor(this, range)
            lo = range(1);
            hi = range(2);
            top = round(lo + this.Difficulty * (hi - lo));
            if top > lo
                n = randi(this.Rng, [lo, top]);
            else
                n = lo;
            end
        end

        function integrate(this, act, dt)
            e = this.Cfg.env;
            c = cos(this.Yaw);
            s = sin(this.Yaw);
            vCmd = [c * act.forward; s * act.forward; act.vertical];
            this.Vel = this.Vel + (vCmd - this.Vel) * (1 - exp(-dt / e.velocityTau));
            if e.windStd > 0
                this.Vel(1:2) = this.Vel(1:2) + e.windStd * randn(this.Rng, 2, 1);
            end
            this.YawRate = this.YawRate + (act.yawRate - this.YawRate) * (1 - exp(-dt / e.yawTau));
            this.Yaw = wrapAngle(this.Yaw + this.YawRate * dt);
            delta = this.Vel * dt;
            this.Pos = this.Pos + delta;
            this.PathLength = this.PathLength + norm(delta);
        end

        function frame = renderFrame(this)
            cam = this.Cfg.env.camera;
            [depth, shade] = renderCamera(this.World, this.Camera, this.Pos, this.Yaw, this.Light);
            this.Depth = min(depth, cam.maxDepth);
            this.SectorDistTrue = sectorMinDepth(this.Depth, this.Cfg.env.sectors.rows, this.Cfg.env.sectors.cols);
            if cam.obsMode == "gray"
                img = shade * this.Contrast + this.Brightness;
            else
                img = this.Depth / cam.maxDepth;
            end
            if cam.noiseStd > 0
                img = img + cam.noiseStd * randn(this.Rng, size(img));
            end
            frame = single(round(min(max(img, 0), 1) * 255) / 255);   % 8-bit camera
        end

        function updatePerception(this)
            e = this.Cfg.env;
            if this.PerceptionSource == "detector"
                this.Percept = detectorPerception(this.Detector, this.Frames, e);
            else
                this.Percept = oraclePerception(this.SectorDistTrue, e);
            end
            if ~isempty(this.Shield)
                vf = this.bodyVelocity();
                [this.Mask, this.ShieldInfo] = this.Shield.computeMask( ...
                    this.Percept.sectorDist, this.Percept.risk, this.Pos(3), vf);
            else
                this.Mask = true(1, numel(this.Actions));
            end
        end

        function a = safeSubstitute(this)
            %SAFESUBSTITUTE  Best safe action according to the reactive planner.
            [~, bearing, dz] = this.goalGeometry();
            a = reactiveAct(this.Planner, this.Percept.sectorDist, bearing, dz, this.Mask);
        end

        function [vf, vl, vu] = bodyVelocity(this)
            c = cos(this.Yaw);
            s = sin(this.Yaw);
            vf = c * this.Vel(1) + s * this.Vel(2);
            vl = -s * this.Vel(1) + c * this.Vel(2);
            vu = this.Vel(3);
        end

        function [dist, bearing, dz] = goalGeometry(this)
            %GOALGEOMETRY  Distance, body-frame bearing (+ = left) and height difference to the goal.
            rel = this.Goal - this.Pos;
            dist = norm(rel);
            bearing = wrapAngle(atan2(rel(2), rel(1)) - this.Yaw);
            dz = rel(3);
        end

        function st = stateVector(this)
            e = this.Cfg.env;
            [dist, bearing, dz] = this.goalGeometry();
            [vf, vl, vu] = this.bodyVelocity();
            vMax = max([this.Actions.forward]);
            vzMax = max(abs([this.Actions.vertical]));
            wMax = max(abs([this.Actions.yawRate]));
            arenaDiag = e.arenaSize * sqrt(2);
            st = [min(dist / arenaDiag, 1)
                sin(bearing)
                cos(bearing)
                min(max(dz / e.ceiling, -1), 1)
                min(max(this.Pos(3) / e.ceiling, 0), 1.5)
                min(max(vf / vMax, -1.5), 1.5)
                min(max(vl / vMax, -1.5), 1.5)
                min(max(vu / vzMax, -1.5), 1.5)
                min(max(this.YawRate / wMax, -1.5), 1.5)
                1 - this.StepCount / e.maxSteps];
        end

        function obs = observation(this)
            st = this.stateVector();
            if this.UseFeatures
                st = [st; double(this.Percept.features(:))];
            end
            cls = char(this.ObsClass);
            obs = {cast(this.Frames, cls), cast(st, cls)};
        end

        function info = makeInfo(this, collision, reached, truncated, clearance, obstacle, requested, shielded, executed)
            [dist, bearing, dz] = this.goalGeometry();
            vf = this.bodyVelocity();
            info = struct( ...
                "collision", collision, ...
                "reached", reached, ...
                "truncated", truncated, ...
                "obstacle", string(obstacle), ...
                "clearance", clearance, ...
                "minClearance", this.MinClearance, ...
                "distance", dist, ...
                "goalBearing", bearing, ...
                "goalDz", dz, ...
                "altitude", this.Pos(3), ...
                "forwardSpeed", vf, ...
                "position", this.Pos, ...
                "yaw", this.Yaw, ...
                "steps", this.StepCount, ...
                "pathLength", this.PathLength, ...
                "startDistance", this.StartDist, ...
                "requestedAction", requested, ...
                "executedAction", executed, ...
                "shielded", shielded, ...
                "interventions", this.Interventions, ...
                "sectorDistTrue", this.SectorDistTrue);
        end

        function randomizeAppearance(this)
            rs = this.Rng;
            if this.Cfg.env.camera.domainRandomization
                az = unif(rs, [0, 2 * pi]);
                el = unif(rs, [deg2rad(30), deg2rad(70)]);
                this.Light = [cos(az) * cos(el); sin(az) * cos(el); sin(el)];
                this.Contrast = unif(rs, [0.85, 1.15]);
                this.Brightness = unif(rs, [-0.08, 0.08]);
            else
                this.Light = [0.4; 0.3; 0.866];
                this.Contrast = 1;
                this.Brightness = 0;
            end
        end

        function logEpisode(this, outcome)
            rec = struct("outcome", string(outcome), "steps", this.StepCount, ...
                "pathLength", this.PathLength, "startDistance", this.StartDist, ...
                "difficulty", this.Difficulty, "interventions", this.Interventions);
            if isempty(this.EpisodeLog)
                this.EpisodeLog = rec;
            else
                this.EpisodeLog(end + 1) = rec;
            end
            this.EpisodeCount = this.EpisodeCount + 1;
        end
    end
end

% ======================================================================= local
function v = unif(rs, range)
v = range(1) + (range(2) - range(1)) * rand(rs);
end

function spec = numericSpec(dim, lo, hi)
% single-precision specs halve the replay-memory footprint when supported
try
    spec = rlNumericSpec(dim, LowerLimit=lo, UpperLimit=hi, DataType="single");
catch
    spec = rlNumericSpec(dim, LowerLimit=lo, UpperLimit=hi);
end
end
