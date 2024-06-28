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
load(init_label_filename);
alpha_set =  10.^[3:-1:-6];
beta_set = 10.^[6:-1:-2];
gamma_set=10.^[2:-1:-6];
s_num_set=[100,200,300,400,500];
s=num_classes;
%%%
view_num=length(samples.data);
newsamples.fea=samples.fea;
for v=1:view_num
    newsamples.data{v}=(samples.data{v})';
end
 DemoCDSL(dataset_name, newsamples, classes, init_labels, Runtimes, ...
    alpha_set, beta_set, gamma_set,s_num_set);



