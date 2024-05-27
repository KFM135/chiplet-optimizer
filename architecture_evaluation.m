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
peak_ops_per_sec_chiplet = [237 249];       % TFLOPs; 64 chiplet = 237, 128 chiplets = 249

ops_per_task = [4, 410, 447, 947, 32];

% Scale factor (because of mapping efficiency; non-linear and other non-GEMM operation). The scale factor is derived from the reference
% throughput got from the actual GPU runs. And then used this scale factor
% in this equation: tasks_per_sec = peak_ops_per_sec_64_chiplet / (scale factor * ops_per_task)

scale_factor = peak_ops_per_sec_A100 ./ (ref_tasks_per_sec .* ops_per_task .* 10^-3);

for i=1:length(peak_ops_per_sec_chiplet)
    tasks_per_sec(i,:) = peak_ops_per_sec_chiplet(i) ./ (ops_per_task .* 10^-3 .* scale_factor);
end
tasks_per_sec_comp = [tasks_per_sec; ref_tasks_per_sec];

figure(1)
t = tiledlayout(1, 3);
t.TileSpacing = "tight";
t.Padding = 'compact';

nexttile(1)

b = bar(tasks_per_sec_comp', 'FaceColor', 'flat', 'EdgeColor', 'flat'); grid on;

% b(1).FaceColor = [0.57,0.75,0.87];
% b(1).LineStyle = 'none';
% b(2).FaceColor = [0.84,0.79,0.77];
% b(2).LineStyle = 'none';
% b(3).FaceColor = [0.78,0.85,0.58];
% b(3).LineStyle = 'none';
legend('60 chiplets', '112 chiplets','GPU', 'NumColumns',3)
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
ax.XTickLabelRotation= 20;
ax.YLim = [min(tasks_per_sec_comp, [], 'all')-10 max(tasks_per_sec_comp, [],"all")+10];
% ax.YScale='log';
ax.YMinorTick='off';
ax.MinorGridLineStyle='none';


%% Energy
joules_per_op_A100 = 400 / (156 * 10 ^12);
joules_per_op = [6.8e-13 7.06e-13 joules_per_op_A100];
ops_per_task = [4, 410, 447, 947, 32];
for i=1:length(joules_per_op)
    tasks_per_joule(i, :) = 1 ./ (joules_per_op(i) .* ops_per_task .* 10^9 .* scale_factor);
end
nexttile(2)
b = bar([1, 2, 3, 4, 5],tasks_per_joule',  'FaceColor', 'flat', 'EdgeColor', 'flat', 'BarWidth',0.8); grid on,
% legend('60 chiplets', '112 chiplets', 'NumColumns', 2)
% ytips1 = b(1).YEndPoints
% xtips1 = b(1).XEndPoints
% labels1 = string(b(1).YData);
% text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom')
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
ax.XLim=[min(ax.XTick)-1 max(ax.XTick)+1];
ax.XTickLabel = model_name;
ax.XTickLabelRotation= 20;
ax.YScale='log';
ax.YMinorTick='off';
ax.MinorGridLineStyle='none';
ax.YLim = [min(tasks_per_joule, [], 'all') - min(tasks_per_joule, [], 'all')/2 max(tasks_per_joule, [], 'all') + max(tasks_per_joule, [], 'all') /2
];
% ax.YTick = [0.01 1];
%ax.YTickLabel = [0.01 1];



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
cost_defect_chips_gpu = cost_defect_die_gpu * num_chiplet;
cost_defect_package_gpu = cost_raw_package_gpu * 1 / (bonding_yield_os^num_chiplet) - 1;
cost_wasted_chips_gpu = (cost_raw_chips_gpu + cost_defect_chips_gpu) * 1 / (bonding_yield_os ^ num_chiplet) - 1;
cost_package_gpu = cost_raw_package_gpu + cost_defect_package_gpu + cost_wasted_chips_gpu;
cost_package_RE_gpu = cost_raw_chips_gpu + cost_defect_chips_gpu + cost_package_gpu;

cost_total_gpu = cost_package_RE_gpu + cost_die_RE_gpu;


%%
% Die cost

num_chiplet_pair = [30 56];
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
        %bonding_yield_chiplet(i,j) = bonding_yield_si(j) ^ sqrt(num_chiplet_pair(i));
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
denom = final_mat(:,3);
x = final_mat ./ denom;

fprintf("Die cost gpu:%d; Package cost gpu:%d\n", cost_die_RE_gpu, cost_package_RE_gpu)
fprintf("Die cost 60 chiplet:%d; Package cost 60 chiplet:%d\n", cost_die_RE_chiplet(1), cost_RE_package_chiplet(1))
fprintf("Die cost 112 chiplet:%d; Package cost 112 chiplet:%d\n", cost_die_RE_chiplet(2), cost_RE_package_chiplet(2))


nexttile(3)
b = bar(x, 'FaceColor', 'flat', 'EdgeColor', 'flat'); grid on;
digits(3)
ytips1 = b(1).YEndPoints;
xtips1 = b(1).XEndPoints;
labels1 = string(vpa(b(1).YData));
labels1 = [labels1(1), labels1(2), labels1(3)];

ytips2 = b(2).YEndPoints;
xtips2 = b(2).XEndPoints;
labels2 = string(vpa(b(2).YData));
labels2 = [labels2(1), labels2(2), labels2(3)];

digits(3)
ytips3 = b(3).YEndPoints;
xtips3 = b(3).XEndPoints;
labels3 = string(vpa(b(3).YData));
labels3 = [labels3(1), labels3(2), labels3(3)];

text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight','bold');
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight','bold');
text(xtips3, ytips3, labels3, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight','bold');
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
ax.XTick=[1 2 3];
% ax.XLim=[min(ax.XTick)-1 max(ax.XTick)+1];
ax.YLim = [min(x,[], "all")- min(x,[], "all")/2 max(x,[], "all")+3];
% ax.YTick = [10^-2 10^0]
% ax.YTickLabel = {'10^{-2}', '1'}

row1 = {'Die cost' 'Package cost' 'Package cost'};
row2 = {'' '(99% bonding yield)' '(100% bonding yield)'};
labelarray = [row1; row2]
labelarray = strjust(pad(labelarray),'right')
tickLabels = strtrim(sprintf('%s\\newline%s\n', labelarray{:}));
ax.XTickLabel = tickLabels
%ax.xticklabels({'die cost \n ', 'package cost \n (99% bonding yield)', 'package cost \n (100% bonding yield)'});
%ax.XTickLabel = {'die cost', 'package cost (99% bonding yield)', 'package cost (100% bonding yield)'};
%ax.TickLabelInterpreter = 'none';
ax.XTickLabelRotation= 25;
ax.YScale='log';
ax.YMinorTick='off';
ax.MinorGridLineStyle='none';




