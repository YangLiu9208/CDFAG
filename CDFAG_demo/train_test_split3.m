function [tr_idx, ts_idx, tr_idx2,ts_idx2]=train_test_split3(set_tr_num,set_tr_num2,source_label,target_label)
    clabel = unique(target_label);
    clabel2 = unique(source_label);
    nclass = length(clabel);
    tr_idx = [];
    ts_idx = [];
    tr_idx2 = [];
    ts_idx2 = [];
     for jj = 1:nclass
            idx_label = find( target_label == clabel(jj));
            idx_label2 = find( source_label == clabel2(jj));
            num = length(idx_label);    
            num2 = length(idx_label2);
            idx_rand = randperm(num);      
            idx_rand2 = randperm(num2);  
            if set_tr_num>length(idx_rand)
                tr_num=length(idx_rand);           
            else
                tr_num=set_tr_num;          
            end
            
            if set_tr_num2>length(idx_rand2)
                tr_num2=length(idx_rand2);           
            else
                tr_num2=set_tr_num2;          
            end
            tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
            ts_idx = [ts_idx; idx_label(idx_rand(tr_num+6:end))];
            tr_idx2 = [tr_idx2; idx_label2(idx_rand2(1:tr_num2))];
            ts_idx2 = [ts_idx2; idx_label2(idx_rand2(tr_num2+6:end))];
      end
end