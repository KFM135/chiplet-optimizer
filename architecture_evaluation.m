clc; close all; clear;


model_name = {'Resnet50v1.5', 'Efficientdet', 'Mask-RCNN', '3D-Unet', 'BERT'};

% Notes: The reference values are taken from this website (https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch) and necessary information are added in below.
% resnet50v1.5 is from PyTorch and run on NVIDIA 1xA100 40GB GPU
% efficientdet is from TensorFlow2 and run on NVIDIA 1xA100 40GB GPU
% mask-RCNN is from PyTorch and run on NVIDIA 1xA100 80GB GPU
% 3D-Unet (UNet-medical) is taken from TensorFlow2 and run on NVIDIA 1xA100 80 GB GPU
% BERT is from TensorFlow and run on NVIDIA 1xA100 40GB GPU on sequence length of 128

ref_tasks_per_sec = [191, 38, 25, 152, 153];
peak_ops_per_sec_A100 = 156;             % TFLOPs
peak_ops_per_sec_chiplet = [237 249 194];       % TFLOPs; 60 chiplet = 237, 112 chiplets = 249, 2 chiplet-3D = 194


ops_per_task = [4, 410, 447, 947, 32];

% Scale factor (probably because of mapping efficiency; and maybe because of the non-linear and other non-GEMM operations. not quite sure
% about this gap). So the scale factor is derived from the reference
% throughput got from the actual GPU runs. And then used this scale factor
% in this equation: tasks_per_sec = peak_ops_per_sec_64_chiplet / (scale factor * ops_per_task)

scale_factor = peak_ops_per_sec_A100 ./ (ref_tasks_per_sec .* ops_per_task .* 10^-3);

for i=1:length(peak_ops_per_sec_chiplet)
    tasks_per_sec(i,:) = peak_ops_per_sec_chiplet(i) ./ (ops_per_task .* 10^-3 .* scale_factor);
end
tasks_per_sec_comp = [tasks_per_sec; ref_tasks_per_sec];

figure(1)
t = tiledlayout(1, 4);
t.TileSpacing = "tight";
t.Padding = 'compact';

nexttile(1)

b = bar(tasks_per_sec_comp', 'FaceColor', 'flat', 'EdgeColor', 'flat'); grid on;

legend('60 chiplets', '112 chiplets','2 chiplets','GPU', 'NumColumns',4)
ylabel('Inferences/sec');
ax = gca; % current axes
ax.FontSize = 12.2;
ax.FontName='Times New Roman';
ax.Title.FontSize=12.2;
ax.Title.FontWeight='normal';
ax.Box='on';
ax.XScale='linear';
ax.XMinorTick='off';
ax.XTick=[1 2 3 4 5];
% ax.XLim=[min(ax.XTick)-1 max(ax.XTick)+1];
ax.XTickLabel= model_name;
ax.XTickLabelRotation= 45;
ax.YLim = [min(tasks_per_sec_comp, [], 'all')-10 max(tasks_per_sec_comp, [],"all")+10];
% ax.YScale='log';
ax.YMinorTick='off';
ax.MinorGridLineStyle='none';


% Energy
joules_per_op_A100 = 400 / (156 * 10 ^12);
joules_per_op = [6.8e-13 7.06e-13 5.52e-13 joules_per_op_A100];
ops_per_task = [4, 410, 447, 947, 32];
for i=1:length(joules_per_op)
    tasks_per_joule(i, :) = 1 ./ (joules_per_op(i) .* ops_per_task .* 10^9 .* scale_factor);
end
nexttile(2)
b = bar([1, 2, 3, 4, 5],tasks_per_joule',  'FaceColor', 'flat', 'EdgeColor', 'flat', 'BarWidth',0.8); grid on,
ylabel('Inferences/joule');
ax = gca; % current axes
ax.FontSize = 12.2;
ax.FontName='Times New Roman';
ax.Title.FontSize=12.2;
ax.Title.FontWeight='normal';
ax.Box='on';
ax.XScale='linear';
ax.XMinorTick='off';
ax.XTick=[1 2 3 4 5];
% ax.XLim=[min(ax.XTick)-1 max(ax.XTick)+1];
ax.XTickLabel = model_name;
ax.XTickLabelRotation= 45;
ax.YScale='log';
ax.YMinorTick='off';
ax.MinorGridLineStyle='none';
ax.YLim = [min(tasks_per_joule, [], 'all') - min(tasks_per_joule, [], 'all')/2 max(tasks_per_joule, [], 'all') + max(tasks_per_joule, [], 'all') /2
];



%% Cost
% Constants (source: Chiplet Actuary paper: https://github.com/Yinxiao-Feng/DAC2022/)
defect_density = 0.09; % 7nm
critical_level = 10;
%critical_level = 6; % 6 for SI
scribe_line = 0.2;
wafer_diameter = 300;
wafer_cost = 9346; 
edge_loss = 5;
os_area_scale_factor = 4;
%package_factor = 2; % factor = 2 as package>900
package_factor = 2;
cost_factor_os = 0.005;
c4_bump_cost_factor = 0.005;
ubump_cost_factor = 0.01;
num_chiplet = 1;
bonding_yield_os = 0.99;
area_scale_factor_si = 1.1;
defect_density_si = 0.06;
bonding_yield_si = [0.99, 1]; 


% GPU cost calculation
% Die cost
gpu_area = 826; %mm2
gpu_area_f = gpu_area + 2 * scribe_line * sqrt(gpu_area) + scribe_line^2;
N_die_total_gpu = pi * ((wafer_diameter/2 - edge_loss)^2)/ gpu_area_f - pi * (wafer_diameter - 2 * edge_loss) / (sqrt(2 * gpu_area_f));
die_yield_gpu = (1 + (defect_density * gpu_area) / (100 * critical_level) ) ^ -critical_level;
N_KGD_gpu = N_die_total_gpu * die_yield_gpu;
cost_raw_die_gpu = wafer_cost / N_die_total_gpu;
cost_KGD_gpu = wafer_cost / N_KGD_gpu;
cost_defect_die_gpu = cost_KGD_gpu - cost_raw_die_gpu;
cost_die_RE_gpu = cost_raw_die_gpu + cost_defect_die_gpu;  % cost_die_RE and cost_KGD are same


% Package cost
package_area_gpu = gpu_area * os_area_scale_factor;
%package_area_gpu = gpu_area * area_scale_factor_si; 
cost_raw_package_gpu = package_area_gpu * cost_factor_os * package_factor;

cost_raw_chips_gpu = (cost_raw_die_gpu + gpu_area * c4_bump_cost_factor) * num_chiplet;
%cost_raw_chips_gpu = gpu_area * c4_bump_cost_factor * num_chiplet;
cost_defect_chips_gpu = cost_defect_die_gpu * num_chiplet;
cost_defect_package_gpu = cost_raw_package_gpu * 1 / (bonding_yield_os^num_chiplet) - 1;
cost_wasted_chips_gpu = (cost_raw_chips_gpu + cost_defect_chips_gpu) * 1 / (bonding_yield_os ^ num_chiplet) - 1;
cost_package_gpu = cost_raw_package_gpu + cost_defect_package_gpu + cost_wasted_chips_gpu;
cost_package_RE_gpu = cost_raw_chips_gpu + cost_defect_chips_gpu + cost_package_gpu;

cost_total_gpu = cost_package_RE_gpu + cost_die_RE_gpu;


%%
% Die cost

%num_chiplet_pair = [60 112];   % change it to num_chiplet
num_chiplet_pair = [30 56 1];
for i=1:length(num_chiplet_pair)
    chiplet_area(i) = 800 / num_chiplet_pair(i);
    chiplet_area_f(i) = chiplet_area(i) + 2 * scribe_line * sqrt(chiplet_area(i)) + scribe_line^2;
    N_die_total_chiplet(i) = pi * ((wafer_diameter/2 - edge_loss)^2)/ chiplet_area_f(i) - pi * (wafer_diameter - 2 * edge_loss) / (sqrt(2 * chiplet_area_f(i)));
    die_yield_chiplet(i) = (1 + (defect_density * chiplet_area(i)) / (100 * critical_level) ) ^ -critical_level;
    N_KGD_chiplet(i) = N_die_total_chiplet(i) * die_yield_chiplet(i);
    cost_raw_die_chiplet(i) = wafer_cost / N_die_total_chiplet(i);
    cost_KGD_chiplet(i) = wafer_cost / N_KGD_chiplet(i);
    cost_defect_die_chiplet(i) = cost_KGD_chiplet(i) - cost_raw_die_chiplet(i);
    cost_die_RE_chiplet(i) = cost_raw_die_chiplet(i) + cost_defect_die_chiplet(i);

    % Package cost
    die_area_tot = 900;  %mm2
    interposer_area(i) = die_area_tot * area_scale_factor_si; 
    package_area_chiplet(i) = interposer_area(i) * os_area_scale_factor;
    package_yield_chiplet(i) = (1 + (defect_density_si * interposer_area(i)) / (100 * critical_level) ) ^ -critical_level;
    interposer_area_f(i) = interposer_area(i) + 2 * scribe_line * sqrt(interposer_area(i)) + scribe_line^2;
    N_package_total(i) = pi * ((wafer_diameter/2 - edge_loss)^2)/ interposer_area_f(i) - pi * (wafer_diameter - 2 * edge_loss) / (sqrt(2 * interposer_area_f(i)));
    cost_interposer(i) = (wafer_cost / N_package_total(i)) + interposer_area(i) * c4_bump_cost_factor;
    cost_substrate(i) = package_area_chiplet(i) * cost_factor_os; %
    cost_3D(i) = 5; % 3D packaging penalty
    cost_raw_package_chiplet(i) = cost_interposer(i) + cost_substrate(i);

    cost_raw_chips_chiplet(i) = cost_raw_die_chiplet(i) * num_chiplet_pair(i) + chiplet_area(i) * ubump_cost_factor + num_chiplet_pair(i)*cost_3D(i);
    cost_defect_chips_chiplet(i) = cost_defect_die_chiplet(i) * num_chiplet_pair(i) + cost_defect_die_chiplet(i) * 2;
    for j = 1:length(bonding_yield_si)
        bonding_yield_chiplet(i,j) = bonding_yield_si(j) ^ (num_chiplet_pair(i)) * bonding_yield_si(j) ^ 2;   % this part (bonding_yield_si ^ 2) is for 3D
        cost_defect_package_chiplet(i,j) = cost_interposer(i) * (1 / (package_yield_chiplet(i) * bonding_yield_chiplet(i,j) * bonding_yield_os) - 1) + cost_substrate(i) * (1/bonding_yield_os - 1); % 
        cost_wasted_chips_chiplet(i,j) = (cost_raw_chips_chiplet(i) + cost_defect_chips_chiplet(i) ) * 1 / (bonding_yield_chiplet(i,j) * bonding_yield_os) - 1;
        cost_RE_package_chiplet(i,j) = cost_raw_chips_chiplet(i) + cost_defect_chips_chiplet(i) + cost_raw_package_chiplet(i) + cost_defect_package_chiplet(i,j) + cost_wasted_chips_chiplet(i,j);

    end
end

disp('cost package chiplet')
disp(cost_RE_package_chiplet')
disp('Cost die chiplet')
disp(cost_die_RE_chiplet)

package_cost_all = [cost_RE_package_chiplet' [cost_package_RE_gpu; cost_package_RE_gpu]];
die_cost_all = [cost_die_RE_chiplet cost_die_RE_gpu];
final_mat = [die_cost_all; package_cost_all];

total_cost_99_bonding_yield = package_cost_all(1,:) + die_cost_all;
total_cost_100_bonding_yield = package_cost_all(2,:) + die_cost_all;
final_mat_total = [die_cost_all; package_cost_all; total_cost_99_bonding_yield; total_cost_100_bonding_yield];
denom = final_mat(:,4);
denom_total = final_mat_total(:,4);
x = final_mat ./ denom;
x_total = final_mat_total ./ denom_total;


%% ---- cost breakdown plot data ------
gpu_cost_breakdown = [cost_raw_die_gpu, cost_defect_die_gpu, cost_raw_chips_gpu, cost_defect_chips_gpu, cost_raw_package_gpu, cost_defect_package_gpu, cost_wasted_chips_gpu];
gpu_cost_breakdown_pct = (gpu_cost_breakdown ./ sum(gpu_cost_breakdown)) .* 100;

chiplet_2_99_cost_breakdown = [cost_raw_die_chiplet(3), cost_defect_die_chiplet(3), cost_raw_chips_chiplet(3), cost_defect_chips_chiplet(3), cost_raw_package_chiplet(3), cost_defect_package_chiplet(3,1), cost_wasted_chips_chiplet(3,1)];
chiplet_2_99_cost_breakdown_pct = (chiplet_2_99_cost_breakdown ./ sum(chiplet_2_99_cost_breakdown)) .* 100;

chiplet_2_100_cost_breakdown = [cost_raw_die_chiplet(3), cost_defect_die_chiplet(3), cost_raw_chips_chiplet(3), cost_defect_chips_chiplet(3), cost_raw_package_chiplet(3), cost_defect_package_chiplet(3,2), cost_wasted_chips_chiplet(3,2)];
chiplet_2_100_cost_breakdown_pct = (chiplet_2_100_cost_breakdown ./ sum(chiplet_2_100_cost_breakdown)) .* 100;

chiplet_60_99_cost_breakdown = [cost_raw_die_chiplet(1), cost_defect_die_chiplet(1), cost_raw_chips_chiplet(1), cost_defect_chips_chiplet(1), cost_raw_package_chiplet(1), cost_defect_package_chiplet(1,1), cost_wasted_chips_chiplet(1,1)];
chiplet_60_99_cost_breakdown_pct = (chiplet_60_99_cost_breakdown ./sum(chiplet_60_99_cost_breakdown)) .* 100;

chiplet_112_99_cost_breakdown = [cost_raw_die_chiplet(2), cost_defect_die_chiplet(2), cost_raw_chips_chiplet(2), cost_defect_chips_chiplet(2), cost_raw_package_chiplet(2), cost_defect_package_chiplet(2,1), cost_wasted_chips_chiplet(2,1)];
chiplet_112_99_cost_breakdown_pct = (chiplet_112_99_cost_breakdown ./sum(chiplet_112_99_cost_breakdown)) .* 100;

chiplet_60_100_cost_breakdown = [cost_raw_die_chiplet(1), cost_defect_die_chiplet(1), cost_raw_chips_chiplet(1), cost_defect_chips_chiplet(1), cost_raw_package_chiplet(1), cost_defect_package_chiplet(1,2), cost_wasted_chips_chiplet(1,2)];
chiplet_60_100_cost_breakdown_pct = (chiplet_60_100_cost_breakdown ./sum(chiplet_60_100_cost_breakdown)) .* 100;

chiplet_112_100_cost_breakdown = [cost_raw_die_chiplet(2), cost_defect_die_chiplet(2), cost_raw_chips_chiplet(2), cost_defect_chips_chiplet(2), cost_raw_package_chiplet(2), cost_defect_package_chiplet(2,2), cost_wasted_chips_chiplet(2,2)];
chiplet_112_100_cost_breakdown_pct = (chiplet_112_100_cost_breakdown ./sum(chiplet_112_100_cost_breakdown)) .* 100;

cost_breakdown_pct_all = [gpu_cost_breakdown_pct; chiplet_2_99_cost_breakdown_pct; chiplet_60_99_cost_breakdown_pct; chiplet_112_99_cost_breakdown_pct; chiplet_2_100_cost_breakdown_pct; chiplet_60_100_cost_breakdown_pct; chiplet_112_100_cost_breakdown_pct]
%cost_breakdown_all = [gpu_cost_breakdown; chiplet_2_99_cost_breakdown; chiplet_60_99_cost_breakdown; chiplet_112_99_cost_breakdown; chiplet_2_100_cost_breakdown; chiplet_60_100_cost_breakdown; chiplet_112_100_cost_breakdown]

%%
nexttile(3)
b = bar(x_total, 'FaceColor', 'flat', 'EdgeColor', 'flat'); grid on;
digits(3)
ytips1 = b(1).YEndPoints
xtips1 = b(1).XEndPoints
labels1 = string(vpa(b(1).YData))
labels1 = [labels1(1), labels1(2), labels1(3), labels1(4), labels1(5)]

ytips2 = b(2).YEndPoints;
xtips2 = b(2).XEndPoints;
labels2 = string(vpa(b(2).YData));
labels2 = [labels2(1), labels2(2), labels2(3), labels2(4), labels2(5)];

digits(3)
ytips3 = b(3).YEndPoints;
xtips3 = b(3).XEndPoints;
labels3 = string(vpa(b(3).YData));
labels3 = [labels3(1), labels3(2), labels3(3), labels3(4), labels3(5)];

text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontWeight','bold', Rotation=90);
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontWeight','bold', Rotation=90);
text(xtips3, ytips3, labels3, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontWeight','bold', Rotation=90);
% b(1).FaceColor = [0.57,0.75,0.87];
% b(1).LineStyle = 'none';
% b(2).FaceColor = [0.84,0.79,0.77];
% b(2).LineStyle = 'none';
% b(3).FaceColor = [0.78,0.85,0.58];
% b(3).LineStyle = 'none';
% legend('60 chiplets', 'GPU')
ylabel({'Normalized', 'cost'});
ax = gca; % current axes
ax.FontSize = 12.2;
ax.FontName='Times New Roman';
ax.Title.FontSize=12.2;
ax.Title.FontWeight='normal';
ax.Box='on';
ax.XScale='linear';
ax.XMinorTick='off';
ax.XTick=[1 2 3 4 5];
% ax.XLim=[min(ax.XTick)-1 max(ax.XTick)+1];
ax.YLim = [min(x,[], "all")- min(x,[], "all")/2 max(x,[], "all")+3];
% ax.YTick = [10^-2 10^0]
% ax.YTickLabel = {'10^{-2}', '1'}

row1 = {'Die cost' 'Integration cost (99% BY)' 'Integration cost (100% BY)' 'Total cost (99% BY)' 'Total cost (100% BY)'};
row2 = {'' '(99% bonding yield)' '(100% bonding yield)' '(99% bonding yield)' '(100% bonding yield)'};
labelarray = [row1; row2];
labelarray = strjust(pad(labelarray),'right');
tickLabels = strtrim(sprintf('%s\\newline%s\n', labelarray{:}));
ax.XTickLabel = row1;
%ax.xticklabels({'die cost \n ', 'package cost \n (99% bonding yield)', 'package cost \n (100% bonding yield)'});
%ax.XTickLabel = {'die cost', 'package cost (99% bonding yield)', 'package cost (100% bonding yield)'};
%ax.TickLabelInterpreter = 'none';
ax.XTickLabelRotation= 45;
ax.YScale='log';
ax.YMinorTick='off';
ax.MinorGridLineStyle='none';

nexttile(4)
% groupLabels = {'Monolithic', '60Chiplet(99% bonding yield)', '112Chiplet(99% bonding yield)', '60Chiplet(100% bonding yield)', '112Chiplet(100% bonding yield)'};

categoryLabels = {'Raw die', 'Defected die', 'Raw chips', 'Defected chips', 'Raw package', 'Defected package', 'Wasted KGD'};
b = bar(cost_breakdown_pct_all, 'stacked', 'FaceColor', 'flat', 'EdgeColor', 'flat', 'BarWidth',0.4); grid on, 
legend(categoryLabels, 'Location', 'best', NumColumns=4);
ylabel({'Cost (%)'});
ax = gca; % current axes
ax.FontSize = 12.2;
ax.FontName='Times New Roman';
ax.Title.FontSize=12.2;
ax.Title.FontWeight='normal';
ax.Box='on';
ax.XScale='linear';
ax.XMinorTick='off';
ax.XTick=[1 2 3 4 5 6 7];
ax.XTickLabelRotation= 45;
ax.YLim = [0,100];
row1 = {'GPU', '2 chiplets (99% BY)', '60 chiplets (99% BY)', '112 chiplets (99% BY)', '2 chiplets (100% BY)','60 chiplets (100% BY)', '112 chiplets (100% BY)'};
row2 = {'' '(99% bonding yield)' '(99% bonding yield)' '(99% bonding yield)' '(100% bonding yield)' '(100% bonding yield)' '(100% bonding yield)'};
labelarray = [row1; row2];
labelarray = strjust(pad(labelarray),'right');
tickLabels = strtrim(sprintf('%s\\newline%s\n', labelarray{:}));
ax.XTickLabel = row1;

