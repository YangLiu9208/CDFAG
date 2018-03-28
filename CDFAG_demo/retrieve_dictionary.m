function [B1,B2] = retrieve_dictionary(target_database,source_database ,para )
Bpath = ['dictionary\dict_' para.target_dataSet '_' num2str(para.numClusters) '.mat'];
Xpath = ['dictionary\rand_patches_' para.target_dataSet '_' num2str(para.nsmp) '.mat'];
Bpath2 = ['dictionary\dict_' para.source_dataSet '_' num2str(para.numClusters2) '.mat'];
Xpath2 = ['dictionary\rand_patches_' para.source_dataSet '_' num2str(para.nsmp2) '.mat'];
if ~para.skip_dic_training,
    try
        load(Xpath);
         load(Xpath2);
    catch
    disp('==================================================');
    fprintf('Rand Sampling...\n');
    disp('==================================================');
        X = rand_sampling(target_database, para.nsmp); %X为随机采样的输入特征向量矩阵，行为特征长度128，列为特征数目
        X2 = rand_sampling(source_database, para.nsmp2); %X为随机采样的输入特征向量矩阵，行为特征长度128，列为特征数目
        save(Xpath, 'X');
        save(Xpath2, 'X2');
    end
    disp('==================================================');
    fprintf('Kmeans...\n');
    disp('==================================================');
     B1=vl_kmeans(X, para.numClusters);
     B2=vl_kmeans(X2, para.numClusters2);
     save(Bpath, 'B1');
     save(Bpath2, 'B2');
  % [B, S, stat] = reg_sparse_coding(X, para.nBases, eye(para.nBases), para.beta, para.gamma, para.num_iters);
   % save(Bpath, 'B', 'S', 'stat');
else
    load(Bpath);
    load(Bpath2);
end
end

