% MetaAnalysis_v0_1_final_fix2.m
% Versión corregida: evita error "YTick values must increase"
clear; close all; clc;

%% Data (embedded)
data = {
'He(2024)_F',   'Elderly_Female', 68, 20.4, 3.6,  36, 14.8, 2.6;
'Li(2024)_F',   'Elderly_Female', 29, 22.4, 3.0,  31, 14.4, 2.8;
'He(2024)_M',   'Elderly_Male',   30, 32.1, 5.3,  19, 24.1, 2.9;
'Li(2024)_M',   'Elderly_Male',   16, 34.1, 5.4,  17, 23.8, 2.2;
'Sepulveda(2025)','Elderly',      22, 24.5, 7.4,  13, 19.8, 5.9;
'He(2024)',     'Elderly',        98, 23.9, 7.0,  55, 18.0, 5.2;
'Li(2024)',     'Elderly',        45, 26.6, 6.9,  48, 17.7, 5.2;
'Hu(2021)',     'Elderly',         5, 29.6, 8.88, 5, 20.0, 3.74;
};

Study = data(:,1);
Subgroup = data(:,2);
nControl = cell2mat(data(:,3));
MeanControl = cell2mat(data(:,4));
SDControl = cell2mat(data(:,5));
nCase = cell2mat(data(:,6));
MeanCase = cell2mat(data(:,7));
SDCase = cell2mat(data(:,8));

N = numel(Study);

MD = MeanControl - MeanCase;
Var_within = (SDControl.^2)./nControl + (SDCase.^2)./nCase;
SE_within = sqrt(Var_within);

T_perstudy = table(Study, Subgroup, nControl, nCase, MeanControl, SDControl, MeanCase, SDCase, MD, SE_within, Var_within, ...
    'VariableNames', {'Study','Subgroup','nControl','nCase','MeanControl','SDControl','MeanCase','SDCase','MD','SE','Var'});

uniqueGroups = unique(Subgroup,'stable');
groupList = [uniqueGroups; {'Overall'}];
Out = struct();

for gi = 1:numel(groupList)
    grp = groupList{gi};
    if strcmp(grp,'Overall')
        idx = 1:N;
    else
        idx = find(strcmp(Subgroup, grp));
    end
    y = MD(idx);
    v = Var_within(idx);
    k = numel(y);
    if k==0, continue; end

    w_fixed = 1./v;
    ybar_fixed = sum(w_fixed .* y) / sum(w_fixed);
    Q = sum(w_fixed .* (y - ybar_fixed).^2);
    C = sum(w_fixed) - sum(w_fixed.^2)/sum(w_fixed);

    tau2_DL = max(0, (Q - (k-1)) / C);
    [theta_DL, var_theta_DL] = pooled_given_tau(y, v, tau2_DL);
    se_theta_DL = sqrt(var_theta_DL);
    ci_DL = theta_DL + norminv([0.025 0.975]) * se_theta_DL;
    wi_DL = 1./(v + tau2_DL);
    weights_pct_DL = 100 * (wi_DL / sum(wi_DL));
    I2 = max(0, (Q - (k-1))/Q * 100);
    dfQ = k - 1;

    upper = max(var(y), 1);
    fun_reml = @(t) restrictedLogLik(max(t,0), y, v);
    try
        tau2_REML = fminbnd(fun_reml, 0, upper*10, optimset('TolX',1e-8,'Display','off'));
        tau2_REML = max(0,tau2_REML);
    catch
        tau2_REML = 0;
    end
    [theta_REML, var_theta_REML] = pooled_given_tau(y, v, tau2_REML);
    se_theta_REML = sqrt(var_theta_REML);
    ci_REML = theta_REML + norminv([0.025 0.975]) * se_theta_REML;
    wi_REML = 1./(v + tau2_REML);
    weights_pct_REML = 100 * (wi_REML / sum(wi_REML));

    name = matlab.lang.makeValidName(grp);
    Out.(name).idx = idx;
    Out.(name).k = k;
    Out.(name).y = y;
    Out.(name).v = v;
    Out.(name).Q = Q;
    Out.(name).df = dfQ;
    Out.(name).I2 = I2;

    Out.(name).tau2_DL = tau2_DL;
    Out.(name).theta_DL = theta_DL;
    Out.(name).se_theta_DL = se_theta_DL;
    Out.(name).ci_DL = ci_DL;
    Out.(name).weights_pct_DL = weights_pct_DL;

    Out.(name).tau2_REML = tau2_REML;
    Out.(name).theta_REML = theta_REML;
    Out.(name).se_theta_REML = se_theta_REML;
    Out.(name).ci_REML = ci_REML;
    Out.(name).weights_pct_REML = weights_pct_REML;
end

% Ensure Overall exists
if ~isfield(Out,'Overall')
    y_all = MD; v_all = Var_within; k_all = numel(y_all);
    w_fixed_all = 1./v_all; ybar_fixed_all = sum(w_fixed_all .* y_all) / sum(w_fixed_all);
    Q_all = sum(w_fixed_all .* (y_all - ybar_fixed_all).^2);
    C_all = sum(w_fixed_all) - sum(w_fixed_all.^2)/sum(w_fixed_all);
    tau2_all_DL = max(0, (Q_all - (k_all-1)) / C_all);
    [theta_DL_all, var_theta_DL_all] = pooled_given_tau(y_all, v_all, tau2_all_DL);
    ci_DL_all = theta_DL_all + norminv([0.025 0.975]) * sqrt(var_theta_DL_all);

    fun_all = @(t) restrictedLogLik(max(t,0), y_all, v_all);
    tau2_all_REML = fminbnd(fun_all, 0, max(var(y_all),1)*10, optimset('TolX',1e-8,'Display','off'));
    tau2_all_REML = max(0,tau2_all_REML);
    [theta_REML_all, var_theta_REML_all] = pooled_given_tau(y_all, v_all, tau2_all_REML);
    ci_REML_all = theta_REML_all + norminv([0.025 0.975]) * sqrt(var_theta_REML_all);

    Out.Overall.k = k_all;
    Out.Overall.y = y_all;
    Out.Overall.v = v_all;
    Out.Overall.Q = Q_all;
    Out.Overall.df = k_all - 1;
    Out.Overall.I2 = max(0, (Q_all - (k_all-1)) / Q_all * 100);

    Out.Overall.tau2_DL = tau2_all_DL;
    Out.Overall.theta_DL = theta_DL_all;
    Out.Overall.ci_DL = ci_DL_all;

    Out.Overall.tau2_REML = tau2_all_REML;
    Out.Overall.theta_REML = theta_REML_all;
    Out.Overall.ci_REML = ci_REML_all;
end

outdir = fullfile(pwd,'outputs');
if ~exist(outdir,'dir'), mkdir(outdir); end
writetable(T_perstudy, fullfile(outdir,'per_study_table.xlsx'));

groupNames = fieldnames(Out);
rows = cell(numel(groupNames),13);
for i=1:numel(groupNames)
    g = groupNames{i};
    S = Out.(g);
    rows{i,1} = g;
    rows{i,2} = S.k;
    rows{i,3} = S.theta_DL;
    rows{i,4} = S.ci_DL(1);
    rows{i,5} = S.ci_DL(2);
    rows{i,6} = S.tau2_DL;
    rows{i,7} = S.theta_REML;
    rows{i,8} = S.ci_REML(1);
    rows{i,9} = S.ci_REML(2);
    rows{i,10} = S.tau2_REML;
    rows{i,11} = S.Q;
    rows{i,12} = S.df;
    rows{i,13} = S.I2;
end
Tsum = cell2table(rows, 'VariableNames', {'Group','k','Theta_DL','CIlo_DL','CIhi_DL','tau2_DL','Theta_REML','CIlo_REML','CIhi_REML','tau2_REML','Q','df','I2'});
writetable(Tsum, fullfile(outdir,'summary_by_group.xlsx'));
save(fullfile(outdir,'MetaAnalysis_results.mat'),'Out','T_perstudy','Tsum');

% verification write if Elderly exists
if isfield(Out,'Elderly')
    S = Out.Elderly;
    Tver = table({'Elderly'}, S.theta_DL, S.ci_DL(1), S.ci_DL(2), S.tau2_DL, 'VariableNames', {'Group','Theta_DL','CIlo_DL','CIhi_DL','tau2_DL'});
    writetable(Tver, fullfile(outdir,'verification_table.xlsx'));
end

% Forest plots (fixed function)
makeForestAll('DL', Out, Study, MD, Var_within, outdir);
makeForestAll('REML', Out, Study, MD, Var_within, outdir);

fprintf('Overall (DL): theta=%.3f, 95%%CI=[%.3f, %.3f], tau2=%.4f, Q=%.3f, I2=%.2f%%\n', ...
    Out.Overall.theta_DL, Out.Overall.ci_DL(1), Out.Overall.ci_DL(2), Out.Overall.tau2_DL, Out.Overall.Q, Out.Overall.I2);
fprintf('Overall (REML): theta=%.3f, 95%%CI=[%.3f, %.3f], tau2=%.4f\n', ...
    Out.Overall.theta_REML, Out.Overall.ci_REML(1), Out.Overall.ci_REML(2), Out.Overall.tau2_REML);
disp(['Outputs in: ' outdir]);

%% Local functions
function out = restrictedLogLik(tau2, y, v)
    if tau2 < 0, tau2 = 0; end
    w = 1./(v + tau2);
    W = sum(w);
    mu_hat = sum(w .* y) / W;
    lnDet = sum(log(v + tau2));
    res = (y - mu_hat);
    ss = sum(w .* res.^2);
    n = length(y);
    out = 0.5 * ( lnDet + ( (n-1) * log(ss) ) - (n-1)*log(n-1) );
    if ~isfinite(out), out = 1e100; end
end

function [theta,var_theta] = pooled_given_tau(y, v, tau2)
    w = 1./(v + tau2);
    theta = sum(w .* y) / sum(w);
    var_theta = 1 / sum(w);
end

function makeForestAll(method, Out, Study, MD, Var_within, outdir)
    y = MD;
    v = Var_within;
    k = numel(y);
    if strcmpi(method,'DL')
        tau2 = Out.Overall.tau2_DL;
        theta = Out.Overall.theta_DL;
        ci = Out.Overall.ci_DL;
    else
        tau2 = Out.Overall.tau2_REML;
        theta = Out.Overall.theta_REML;
        ci = Out.Overall.ci_REML;
    end
    wi = 1./(v + tau2);
    weights_pct = 100 * (wi / sum(wi));

    fig = figure('Visible','off','Units','pixels','Position',[100 100 1000 600]);
    ax = axes('Parent', fig);
    hold(ax,'on');

    % We'll plot with increasing YTick (1..k) and set YDir reverse so top=1
    ypos = 1:k;
    % The study i should be placed at position ypos(i) but to display first study at top, reverse labels
    % We'll plot each study at y = ypos(i) and then reverse YDir and set labels accordingly flipped.
    for i=1:k
        xi = y(i);
        sei = sqrt(v(i));
        ci_low = xi - 1.96 * sei;
        ci_high = xi + 1.96 * sei;
        plot(ax, [ci_low, ci_high], [ypos(i), ypos(i)], 'k-', 'LineWidth', 1.4);
        sz = max(6, 40 * weights_pct(i) / max(weights_pct));
        scatter(ax, xi, ypos(i), sz, 'k', 'filled');
    end

    % diamond below the list
    patch_x = [ci(1), theta, ci(2), theta];
    h = 0.6;
    % place diamond at y = 0.5 (below y=1)
    patch_y = [0.5-h/2, 0.5, 0.5+h/2, 0.5];
    patch(ax, patch_x, patch_y, 'k', 'FaceAlpha', 0.4, 'EdgeColor', 'k');

    % aesthetics
    ylim([0 k+1]);
    xmargin = max(1, max(abs(y)) * 0.2);
    xlim([min(y - 3*sqrt(v)) - xmargin, max(y + 3*sqrt(v)) + xmargin]);

    % set increasing yticks and labels, then reverse direction so top shows first label
    set(ax, 'YTick', ypos, 'YTickLabel', Study, 'YDir', 'reverse');

    xlabel(ax, 'Mean Difference (Control - Case)');
    title(ax, ['Forest plot (' method ')']);

    % textual MD [CI] and weights to right
    xlims = xlim(ax);
    x_text = xlims(2) - 0.02 * range(xlims);
    for i=1:k
        txt = sprintf('%.2f [%.2f, %.2f]  (%.1f%%)', y(i), y(i)-1.96*sqrt(v(i)), y(i)+1.96*sqrt(v(i)), weights_pct(i));
        text(ax, x_text, ypos(i), txt, 'HorizontalAlignment','right', 'FontSize',9);
    end

%     % --- Save figure using exportgraphics (MATLAB R2024b) ---
% fname = fullfile(outdir, ['Forest_' method]);
% 
% % PNG (raster) at 300 dpi
% exportgraphics(fig, [fname '.png'], 'Resolution', 300);
% 
% % PDF (vector)
% exportgraphics(fig, [fname '.pdf'], 'ContentType', 'vector');
% 
% % EPS (vector). exportgraphics writes EPS when extension .eps and ContentType 'vector'
% exportgraphics(fig, [fname '.eps'], 'ContentType', 'vector');
% 
% close(fig);

% --- Save figure using exportgraphics with image output for speed/no warnings ---
fname = fullfile(outdir, ['Forest_' method]);

% PNG (raster) at 300 dpi
exportgraphics(fig, [fname '.png'], 'Resolution', 300, 'ContentType','image');

% PDF as embedded image (better performance, no vectorization)
exportgraphics(fig, [fname '.pdf'], 'ContentType','image', 'Resolution',300);

% EPS as raster (note: EPS will contain bitmap)
exportgraphics(fig, [fname '.eps'], 'ContentType','image', 'Resolution',300);

close(fig);

end
