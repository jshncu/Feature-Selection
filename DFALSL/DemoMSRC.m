function DemoMSRC()
maxNumCompThreads(16)
datasetname = 'MSRC_v1';
filename = sprintf('%s.mat', datasetname);
load(filename);
[~, classes] = max(tag, [], 2);
classes = double(classes - min(classes) + 1);
numView = length(X.data);
for i = 1 : numView
    inXCell{i} = X.data{i}';
end
times_clustering = 20;
[dim, nSmp] = size(inXCell{1});
filename = sprintf('init_%s.mat', datasetname);
if exist(filename, 'file')
    load(filename);
else
    % initial clustering labels for k-means
    init_labels = ceil(length(unique(classes)) .* rand(length(classes), times_clustering));
    save(filename, 'init_labels');
end
w = warning ('off','all');
alpha_set = 10.^[4];
beta_set = 10.^[6];
mu_set = 10.^[-4];
r = 2;
m_set = [0.2];
d = 0;
for v = 1 : length(inXCell)
	d = d + size(inXCell{v}, 1);
end
if nSmp < d
	m_set = ceil(m_set * nSmp); 
else
	m_set = ceil(m_set * d);
end
s_num_set = [100:100:500];
DemoDFALSL(datasetname, inXCell, classes, init_labels, times_clustering, ...
	alpha_set, beta_set, mu_set, r, m_set, s_num_set);
warning(w);
end

