classdef SafetyShield < handle
    %SAFETYSHIELD  Collision-avoidance shield between the DQN and the flight controller.
    %
    %   The shield uses per-sector obstacle distances and the crash-risk
    %   probability from the CNN detector (or ground truth in "oracle" mode).
    %   An action is UNSAFE if
    %     1. distance: an obstacle in the sectors it flies into is closer than
    %        uavRadius + margin + speed * reactionTime;
    %     2. time-to-collision: distance / smoothed closing speed < ttcThreshold
    %        in those sectors (detects moving obstacles early);
    %     3. learned risk: crash probability >= riskThreshold blocks fast
    %        forward flight;
    %     4. geofence: no climbing near the ceiling, no descending near the ground.
    %   If an obstacle closes in faster than the UAV's own motion explains,
    %   hovering is also unsafe (the UAV must evade). If everything is
    %   blocked, the action with the largest clearance stays allowed.
    %
    %   The environment applies the shield as a *post-posed shield*: an unsafe
    %   action chosen by the agent is replaced by the best safe action and a
    %   small penalty is given, so the agent learns to avoid unsafe choices.

    properties
        Actions
        Reqs
        Margin
        ReactionTime
        TtcThreshold
        RiskThreshold
        MinAltitude
        CeilingMargin
        Smoothing
        ThreatSpeed
        UavRadius
        Dt
        Ceiling
        HoverIdx
        FastForward
        Center
        NumSectors
        Prev = []
        Closing
    end

    methods
        function this = SafetyShield(cfg, actions)
            s = cfg.shield;
            e = cfg.env;
            this.Actions = actions;
            this.Reqs = actionSectorRequirements(e.sectors.rows, e.sectors.cols, actions);
            this.Margin = s.margin;
            this.ReactionTime = s.reactionTime;
            this.TtcThreshold = s.ttcThreshold;
            this.RiskThreshold = s.riskThreshold;
            this.MinAltitude = s.minAltitude;
            this.CeilingMargin = s.ceilingMargin;
            this.Smoothing = s.closingSmoothing;
            this.ThreatSpeed = s.threatSpeed;
            this.UavRadius = e.uavRadius;
            this.Dt = e.dt * e.actionRepeat;
            this.Ceiling = e.ceiling;
            this.HoverIdx = find([actions.forward] == 0 & [actions.vertical] == 0 & [actions.yawRate] == 0, 1);
            this.FastForward = [actions.forward] >= 1.5;
            this.Center = sectorIndices(e.sectors.rows, e.sectors.cols, "mid", "center");
            this.NumSectors = e.sectors.rows * e.sectors.cols;
            this.reset();
        end

        function reset(this)
            this.Prev = [];
            this.Closing = zeros(this.NumSectors, 1);
        end

        function d = requiredClearance(this, speed)
            d = this.UavRadius + this.Margin + speed * this.ReactionTime;
        end

        function [mask, info] = computeMask(this, sectorDist, risk, altitude, forwardSpeed)
            %COMPUTEMASK  Boolean 1 x numActions mask of safe actions.
            d = double(sectorDist(:));
            if ~isempty(this.Prev)
                raw = (this.Prev - d) / this.Dt;
                this.Closing = this.Smoothing * this.Closing + (1 - this.Smoothing) * raw;
            end
            this.Prev = d;
            ttc = inf(size(d));
            closingNow = this.Closing > 0.2;
            ttc(closingNow) = d(closingNow) ./ this.Closing(closingNow);

            nA = numel(this.Actions);
            mask = true(1, nA);
            clearance = inf(1, nA);
            reasons = strings(1, nA);
            for i = 1:nA
                R = this.Reqs{i};
                for r = 1:numel(R)
                    dmin = min(d(R(r).sectors));
                    clearance(i) = min(clearance(i), dmin);
                    if dmin < this.requiredClearance(R(r).speed)
                        mask(i) = false;
                        reasons(i) = reasons(i) + "distance ";
                    end
                    if R(r).speed > 0.5 && min(ttc(R(r).sectors)) < this.TtcThreshold
                        mask(i) = false;
                        reasons(i) = reasons(i) + "ttc ";
                    end
                end
            end

            if risk >= this.RiskThreshold
                mask(this.FastForward) = false;
                reasons(this.FastForward) = reasons(this.FastForward) + "risk ";
            end

            vert = [this.Actions.vertical];
            tooHigh = vert > 0 & altitude > this.Ceiling - this.CeilingMargin;
            tooLow = vert < 0 & altitude < this.MinAltitude;
            mask(tooHigh | tooLow) = false;
            reasons(tooHigh) = reasons(tooHigh) + "ceiling ";
            reasons(tooLow) = reasons(tooLow) + "ground ";

            % moving threat: closing faster than our own forward motion explains
            c = this.Center;
            own = max(forwardSpeed, 0);
            threat = any(ttc(c) < this.TtcThreshold & this.Closing(c) > own + this.ThreatSpeed);
            if threat && ~isempty(this.HoverIdx)
                mask(this.HoverIdx) = false;
                clearance(this.HoverIdx) = min(d(c));
                reasons(this.HoverIdx) = reasons(this.HoverIdx) + "threat ";
            end

            fallback = false;
            if ~any(mask)
                fallback = true;
                score = clearance;
                score(tooHigh | tooLow) = -inf;
                [~, best] = max(score);
                mask(best) = true;
            end
            info = struct("ttc", ttc, "closing", this.Closing, "clearance", clearance, ...
                "threat", threat, "fallback", fallback, "reasons", reasons);
        end
    end
end
