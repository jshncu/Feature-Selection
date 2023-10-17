clc;clear;

dataset_name = 'MSRC_v1';
load('MSRC_v1.mat');

[~,class_num] = size(tag);  
[num, diemnsion] = size(D); 
k = class_num;
label = zeros(num, 1);
for i = 1:num
   label(i)=find(tag(i,:)==1); 
end
Y=label;
classes = Y;
samples = X; 

assert(min(classes) == 1);
num_classes = length(unique(classes));
init_label_filename = sprintf('init_%s.mat', dataset_name);
num_samples = size(tag, 1);
Runtimes = 20;
if exist(init_label_filename, 'file')
    load(init_label_filename);
else
    init_labels = ceil(num_classes .* rand(num_samples, Runtimes));
    save(init_label_filename, 'init_labels');
end
tmax = 20;

alpha_set = 10.^(2);
beta_set = 10.^(-1);
gamma_set=10.^(3);
t_set=10.^(-3);

r1_set=2;
s_num_set = 100:100:500;
m_set = num_classes;
%%%init_S_MSRC_v1.mat from init_create_S_and_S_bar.m
filename = sprintf('init_S_nor_%s.mat', dataset_name);

if exist(filename, 'file')
    load(filename);
else
    sprintf('Cannot find the file init_S_nor_%s.mat, which is calculated by caculate_jaccardSim_for_p_q.m',dataset_name);
end

DemoMLCL(dataset_name, samples, classes, init_labels, Runtimes, ...
    alpha_set,beta_set,gamma_set,t_set,m_set, s_num_set,r1_set,S,S_bar)
