function DemoMLCL(dataset_name, samples, classes, init_labels, Runtimes, ...
    alpha_set,beta_set,gamma_set,t_set,m_set, s_num_set,r1_set,S,S_bar)

method = 'MLCL';
foldname = 'results_for_MLCL';
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
foldname = sprintf('%s/%s', foldname, dataset_name);
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
foldname = sprintf('%s/%s', foldname, method);
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
foldname = sprintf('%s/%s_v2', foldname, 'kmeans');
if ~exist(foldname, 'dir')
    mkdir(foldname);
end
NMI1 = zeros(Runtimes, length(s_num_set));
Acc = zeros(Runtimes, length(s_num_set));
CCost = zeros(Runtimes, length(s_num_set));
LCost = zeros(Runtimes, 1);
num_classes = length(unique(classes));
Num_View=length(samples.data); 
[n,d]=size(samples.fea);
G=cell(1,Num_View);
for t = t_set
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'HeatKernel';
    options.t = t;
for v=1:Num_View
     G{v} = constructW(samples.data{v},options);
end

    for alpha = alpha_set 
        for beta = beta_set
            for gamma=gamma_set
                for r1=r1_set              
                    filename = sprintf('%s/%s_kmeans_rdim%f_alpha%f_beta%f_gamma%f_r1%f_t.mat', ...
                            foldname, method,alpha,beta,gamma,r1,t);
%                         if(exist(filename, 'file'))
%                             continue;
%                         end
                        begin = tic;
                        W=MLCL(samples.data,r1,r1,alpha,beta,gamma,m_set,S,S_bar,G);
                        LCost = toc(begin);
                        for i = 1 : Runtimes
                            fprintf('%f, %f, %f, %f,%f %d-th repeation\n', alpha, beta,gamma,r1,i);
                            W1=[];
                            for v=1:Num_View   
                                for h = 1:size(samples.data{v},2)
                                    W1 = [W1 norm(W{v}(h,:),2)];
                                end
                            end
                            score=W1;
                            [res, idx] = sort(score, 'descend');             
                            for s_num_ind = 1 : length(s_num_set)
                                s_num = s_num_set(s_num_ind);
                                X = samples.fea(:,idx(1 : s_num));
                                begin = tic;
                                [vObjValues, labels] = Kmeans(X', num_classes, init_labels(:, i)', 100);
                                CCost(i, s_num_ind) = toc(begin);
                                NMI1(i, s_num_ind) = nmi1(classes, labels);
                                res = bestMap(classes, labels);
                                if size(classes) ~= size(res)
                                    res = res';
                                end
                                Acc(i, s_num_ind) = length(find(classes == res)) / length(classes);
                                fprintf('NMI1: %f, Acc: %f\n', NMI1(i, s_num_ind), Acc(i, s_num_ind));
                            end
                            save(filename, 'LCost', 'CCost', 'NMI1', 'Acc');
                        end         
                end
            end
        end
    end
end

