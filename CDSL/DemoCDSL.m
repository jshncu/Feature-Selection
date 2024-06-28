function DemoCDSL(dataset_name, samples, classes, init_labels, Runtimes, ...
    alpha_set, beta_set,gamma_set,s_num_set)
method = 'CDSL';
foldname = 'results';
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
num_classes = length(unique(classes));
[n,~] = size(samples.fea);
% s_set=[0.3*n:0.1*n:0.5*n];
s_set=0.3*n;
for s = s_set
	s=floor(s);
	G_initialization= sprintf('/G_init_mu0.001_s%d.mat',s);
	if exist(G_initialization, 'file')
		load(G_initialization);
		fprintf('Loading G initialization ok \n ');  
      else
		fprintf('G not exist \n');
		pause;
	end
    for alpha = alpha_set
        for beta = beta_set
            for gamma = gamma_set
                filename = sprintf('%s/%s_kmeans_rdim_alpha%f_beta%f_gamma%f_s%d.mat', ...
                    foldname, method, alpha, beta, gamma, s);
                
                if exist(filename,'file')
			        fprintf('%s\n', filename);
                    continue;
                end
                fprintf(' %f, %f,%f,%d-repeation\n',  alpha, beta, gamma,s);
                begin = tic;
                [W] = CDSL(samples.data,samples.fea',s,alpha,beta,gamma,G,num_classes,2); 
                LCost = toc(begin);
                [~,view_num]=size(W);
                %score = sum(W .* W, 2);
                W1=[];
                for v=1:view_num
                    [dv,~]=size(W{v});
                    for k = 1:dv
                        W1 = [W1 norm(W{v}(k,:),2)];
                    end
                end
                score=W1;
                [~, idx] = sort(score, 'descend');
                for i = 1 :  Runtimes  
                    for s_num_ind = 1 : length(s_num_set)
                        s_num = s_num_set(s_num_ind);
                        X = samples.fea(:,idx(1 : s_num));
                        
                        begin = tic; 
                        [~, labels] = Kmeans(X', num_classes, init_labels(:, i)', 100);   
                        
                        CCost(i, s_num_ind) = toc(begin);
                        NMI1(i, s_num_ind) = nmi1(classes, labels);
                        res = bestMap(classes, labels);
                        if size(classes) ~= size(res)
                            res = res';
                        end
                        Acc(i, s_num_ind) = length(find(classes == res)) / length(classes);
                        
                    end
                end
                save(filename, 'LCost', 'CCost', 'NMI1', 'Acc');
            end
        end
    end
end

end
