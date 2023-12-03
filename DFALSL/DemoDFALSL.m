function DemoDFALSL(dataset_name, inXCell, classes, init_labels, Runtimes, ...
    alpha_set, beta_set, mu_set, r, m_set, s_num_set)     %s_num_set:the number of samples.   m:DictSize        
method = 'DFALSL'; % with the lambda set to be 0.001 in spams

addpath('test_release');
addpath('src_release');
addpath('build');
setenv('MKL_NUM_THREADS','1')
setenv('MKL_SERIAL','YES')
setenv('MKL_DYNAMIC','NO')

foldname = '../result'; 
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
foldname = sprintf('%s/%s', foldname, dataset_name);
if ~exist(foldname, 'dir')
    mkdir(foldname);
end

foldname = sprintf('%s/%s_Sylvester', foldname, method); 
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
foldname = sprintf('%s/%s', foldname, 'kmeans');
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
NMI1 = zeros(Runtimes, length(s_num_set));
Acc = zeros(Runtimes, length(s_num_set));
CCost = zeros(Runtimes, length(s_num_set));
LCost = 0;
ICost = 0;
% labels = cell(length(s_num_set), 1);
num_classes = length(unique(classes));

numViews = length(inXCell);
nSmp = size(inXCell, 2);
ds = zeros(numViews, 1);
samples = [];
for i = 1 : numViews
    ds(i) = size(inXCell{i}, 1);
    samples = [samples; inXCell{i}];
end

[d, n] = size(samples);
% Lap Matrix
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'HeatKernel';
options.t = 1;
Smatrix = constructW(samples',options);
S = full(bsxfun(@rdivide,Smatrix,sum(Smatrix,2)));
% S = (ones(n) - eye(n)) / (n - 1);
s_num_set
for m = m_set
    if ~exist(init_U, 'file')
        lambda = 0.001;
		begin = tic;
        [U, A] = Initilization(full(samples), lambda,m);
		ICost = toc(begin);
        save(init_U, 'U', 'lambda');
    else
        load(init_U);
    end
    for alpha = alpha_set
        for beta1 = beta_set
            for mu = mu_set
                filename = sprintf('%s/%s_kmeans_rdim%d_alpha%f_beta%f_mu%f.mat', ...
                    foldname, method, m, alpha, beta1, mu)
                 fprintf('%f, %f, %f, %d\t', alpha, beta1, mu, m);
                 params.alpha = alpha;
                 params.beta = beta1;
                 params.mu = mu;
                 params.r = r;
                 params.mode = 'Sylvester';
                 begin = tic;
                 [W, obj] = DFALSL(samples, ds, U, A, S, params);
                 LCost = toc(begin);
                 score = sum(W .* W, 2);				%W:weight
                 for i = 1 : Runtimes
                     [res, idx] = sort(score, 'descend');		%descend sort
                     for s_num_ind = 1 : length(s_num_set)
                         s_num = s_num_set(s_num_ind);
                         X = samples(idx(1 : s_num), :);
                         begin = tic;
                         [vObjValues, labels] = Kmeans(X, num_classes, init_labels(:, i)', 100);
                         CCost(i, s_num_ind) = toc(begin);
                         NMI1(i, s_num_ind) = nmi1(classes, labels);
                         res = bestMap(classes, labels);
                         if size(classes) ~= size(res)
                             res = res';
                         end
                         Acc(i, s_num_ind) = length(find(classes == res)) / length(classes);
                     end
                 end
                 fprintf('%f\t', mean(NMI1));
                 fprintf('\n');
                 fprintf('%f\t', mean(Acc));
                 fprintf('\n');
                 save(filename, 'LCost', 'CCost', 'ICost', 'NMI1', 'Acc', 'obj', 'score');
            end
        end
     end
end
