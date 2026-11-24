classdef SimViewer < handle
    %SIMVIEWER  Live 2-panel view of the simulation: onboard camera + top-down map.
    %
    %   viewer = SimViewer();
    %   viewer.newEpisode(env);                 % after env.reset
    %   frame = viewer.update(env, "forward");  % after every env.step (returns RGB frame)
    %   viewer.closeFigure();
    %
    %   Camera panel: red cells = image sectors in which the (detected or true)
    %   nearest obstacle is closer than the occupancy threshold.
    %   Map panel: dark grey = pillars taller than the UAV, light grey = pillars
    %   it can fly over, red = moving obstacles, green ring = goal, orange = path.

    properties
        Fig
        AxCam
        AxMap
        CamImage
        SectorPatches
        MoverRects
        TrailLine
        UavPatch
        FovLines
        InfoText
    end

    methods
        function this = SimViewer(titleText)
            if nargin < 1
                titleText = "UAV visual navigation - MATLAB simulation";
            end
            this.Fig = figure(Name=titleText, NumberTitle="off", Color="w", Position=[80 80 1080 540]);
            tl = tiledlayout(this.Fig, 1, 2, TileSpacing="compact", Padding="compact");
            this.AxCam = nexttile(tl);
            this.AxMap = nexttile(tl);
        end

        function newEpisode(this, env)
            e = env.Cfg.env;
            % ---------------------------------------------------------- camera
            cla(this.AxCam);
            img = env.Frames(:, :, end);
            this.CamImage = image(this.AxCam, repmat(img, 1, 1, 3));
            axis(this.AxCam, "image");
            axis(this.AxCam, "off");
            hold(this.AxCam, "on");
            title(this.AxCam, sprintf("Onboard camera (red: obstacle < %.0f m)", e.sectors.occupancyThreshold));
            [H, W] = size(img);
            rows = e.sectors.rows;
            cols = e.sectors.cols;
            re = linspace(0.5, H + 0.5, rows + 1);
            ce = linspace(0.5, W + 0.5, cols + 1);
            this.SectorPatches = gobjects(rows * cols, 1);
            for r = 1:rows
                for c = 1:cols
                    k = (r - 1) * cols + c;
                    this.SectorPatches(k) = patch(this.AxCam, [ce(c) ce(c+1) ce(c+1) ce(c)], ...
                        [re(r) re(r) re(r+1) re(r+1)], [1 0.1 0.1], FaceAlpha=0, ...
                        EdgeColor=[1 1 1], EdgeAlpha=0.35);
                end
            end

            % ------------------------------------------------------------- map
            ax = this.AxMap;
            cla(ax);
            hold(ax, "on");
            L = env.World.size;
            axis(ax, "equal");
            xlim(ax, [0 L]);
            ylim(ax, [0 L]);
            box(ax, "on");
            title(ax, "Top-down map");
            xlabel(ax, "x [m]");
            ylabel(ax, "y [m]");
            for i = 1:size(env.World.pillars, 1)
                p = env.World.pillars(i, :);
                col = [0.33 0.33 0.36];
                if p(4) < e.ceiling
                    col = [0.62 0.62 0.66];               % can be over-flown
                end
                rectangle(ax, Position=[p(1)-p(3), p(2)-p(3), 2*p(3), 2*p(3)], Curvature=[1 1], ...
                    FaceColor=col, EdgeColor="none");
            end
            g = env.Goal;
            gr = e.goalRadius;
            rectangle(ax, Position=[g(1)-gr, g(2)-gr, 2*gr, 2*gr], Curvature=[1 1], ...
                EdgeColor=[0.1 0.6 0.2], LineWidth=2);
            plot(ax, env.Start(1), env.Start(2), "o", MarkerFaceColor=[0.2 0.4 0.9], MarkerEdgeColor="none");
            nM = size(env.World.moverPos, 1);
            this.MoverRects = gobjects(nM, 1);
            for j = 1:nM
                this.MoverRects(j) = rectangle(ax, Position=moverBox(env.World, j), Curvature=[1 1], ...
                    FaceColor=[0.85 0.2 0.2], EdgeColor="none");
            end
            this.TrailLine = plot(ax, env.Pos(1), env.Pos(2), "-", Color=[1 0.55 0], LineWidth=1.6);
            this.FovLines = plot(ax, nan(2, 2), nan(2, 2), "-", Color=[0.55 0.68 0.95]);
            this.UavPatch = patch(ax, nan(1, 3), nan(1, 3), [0.2 0.4 0.9], EdgeColor="none");
            this.InfoText = text(ax, 1, L - 1, "", FontSize=9, VerticalAlignment="top", ...
                Interpreter="none", BackgroundColor="w", Margin=2);
            this.update(env, "");
        end

        function frame = update(this, env, actionName)
            e = env.Cfg.env;
            this.CamImage.CData = repmat(env.Frames(:, :, end), 1, 1, 3);
            d = env.Percept.sectorDist;
            thr = e.sectors.occupancyThreshold;
            for k = 1:numel(this.SectorPatches)
                alpha = 0;
                if d(k) < thr
                    alpha = 0.15 + 0.5 * (1 - d(k) / thr);
                end
                this.SectorPatches(k).FaceAlpha = alpha;
            end

            for j = 1:numel(this.MoverRects)
                this.MoverRects(j).Position = moverBox(env.World, j);
            end
            this.TrailLine.XData = env.Trail(1, :);
            this.TrailLine.YData = env.Trail(2, :);
            x = env.Pos(1);
            y = env.Pos(2);
            yaw = env.Yaw;
            ang = [0 2.5 -2.5];
            rad = [1.4 0.9 0.9];
            this.UavPatch.XData = x + rad .* cos(yaw + ang);
            this.UavPatch.YData = y + rad .* sin(yaw + ang);
            half = env.Camera.fov / 2;
            R = e.camera.maxDepth;
            this.FovLines(1).XData = [x, x + R * cos(yaw + half)];
            this.FovLines(1).YData = [y, y + R * sin(yaw + half)];
            this.FovLines(2).XData = [x, x + R * cos(yaw - half)];
            this.FovLines(2).YData = [y, y + R * sin(yaw - half)];

            dist = env.goalGeometry();
            clearance = worldClearance(env.World, env.Pos);
            txt = sprintf("step %d   action: %s\ngoal %.1f m   clearance %.2f m   altitude %.1f m", ...
                env.StepCount, actionName, dist, clearance, env.Pos(3));
            blocked = [env.Actions(~env.Mask).name];
            if ~isempty(blocked)
                txt = txt + newline + "shield blocks: " + strjoin(blocked, ", ");
            end
            this.InfoText.String = txt;
            drawnow;
            if nargout > 0
                f = getframe(this.Fig);
                frame = f.cdata;
            end
        end

        function closeFigure(this)
            if ~isempty(this.Fig) && isvalid(this.Fig)
                close(this.Fig);
            end
        end
    end
end

function pos = moverBox(world, j)
p = world.moverPos(j, :);
r = world.moverRadius(j);
pos = [p(1) - r, p(2) - r, 2 * r, 2 * r];
end
