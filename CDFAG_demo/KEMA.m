function [ X1_F,X2_F,X1_TF,X2_TF,Y1,Y2,YT1,YT2]=KEMA(XT1,XT2,Xtemp1,Xtemp2,Ytemp1,Ytemp2,YT1,YT2,NF,N_source,N_target,mu)
disp('  Mapping with the RBF kernel ...');
%数据初始化
% parameters
        NF = NF;    %特征维数
        N=N;        %每类训练样本数目
        mu = mu;    %(1-mu)*L  + mu*(Ls)
%         d1=Pca_dimension;
%         d2=Pca_dimension;
%         d=Pca_dimension*2;
        options.graph.nn = 10;  %KNN graph number of neighbors
        r1 = []; rT1 = []; r2 = []; rT2 = [];   
% 1) Data in a block diagonal matrix
%         XT1 = X1(1:2:end,:)';
%         YT1 = Y1(1:2:end,:)';
%         T = size(XT1,2)/2;        
%         Xtemp1 = X1(2:2:end,:);
%         Ytemp1 = Y1(2:2:end,:);        
%         XT2 = X2(1:2:end,:)';
%         YT2 = Y2(1:2:end,:)';        
%         Xtemp2 = X2(2:2:end,:);
%         Ytemp2 = Y2(2:2:end,:);
         t1 = size(XT1,2)/2;  
         t2 = size(XT2,2)/2; 
        [X1 Y1 U1 Y1U indices] = ppc(Xtemp1,Ytemp1,N_source,1);
        [X2 Y2 U2 Y2U indices] = ppc(Xtemp2,Ytemp2,N_target,1);        
        X1 = X1';
        X2 = X2';
        U1 = U1';
        U2 = U2'; 
%         U1 = U1(1:2:end,:)';
%         U2 = U2(1:2:end,:)';        
        clear *temp*        
        Y1U = zeros(size(U1,2),1);
        Y2U = zeros(size(U2,2),1);        
        ncl = numel(unique(Y1));        
        Y = [Y1;Y1U;Y2;Y2U];
        YT = [YT1;YT2];
        [d1 n1] = size(X1);
        [d2 n2] = size(X2);        
        [temp,u1] = size(U1);
        [temp,u2] = size(U2);       
        n = n1+n2+u1+u2;
        d = d1+d2;        
        n1=n1+u1;
        n2=n2+u2;        
        [dT1 T1] = size(XT1);
        [dT2 T2] = size(XT2);     
        dT = dT1+dT2;
% 2) Compute RBF kernels
        sigma1 =  15*mean(pdist([X1]'));
        K1 = kernelmatrix('rbf',[X1,U1],[X1,U1],sigma1);
        sigma2 =  15*mean(pdist([X2]'));
        K2 = kernelmatrix('rbf',[X2,U2],[X2,U2],sigma2);
        K = blkdiag(K1,K2);
        KT1 = kernelmatrix('rbf',[X1,U1],XT1,sigma1);
        KT2 = kernelmatrix('rbf',[X2,U2],XT2,sigma2);
%         Z = blkdiag(X1,[X2,U2]); % (d1+d2) x (n1+n2)
% 2) graph Laplacians         
        G1 = buildKNNGraph([X1,U1]',options.graph.nn,1);
        G2 = buildKNNGraph([X2,U2]',options.graph.nn,1);
        W = blkdiag(G1,G2);
        W = double(full(W));
        clear G*    
% Class Graph Laplacian
        Ws = repmat(Y,1,length(Y)) == repmat(Y,1,length(Y))'; Ws(Y == 0,:) = 0;Ws(:,Y == 0) = 0; Ws = double(Ws);
        Wd = repmat(Y,1,length(Y)) ~= repmat(Y,1,length(Y))';Wd(Y == 0,:) = 0; Wd(:,Y == 0) = 0; Wd = double(Wd);       
        Sws = sum(sum(Ws));
        Sw = sum(sum(W));
        Ws = Ws/Sws*Sw;
        Swd = sum(sum(Wd));
        Wd = Wd/Swd*Sw;        
%         figure(1),
%         imagesc(W);
%         figure(2),
%         imagesc(Ws);
%         figure(3),
%         imagesc(Wd);
        Ds = sum(Ws,2); Ls = diag(Ds) - Ws;
        Dd = sum(Wd,2); Ld = diag(Dd) - Wd;
        D = sum(W,2); L = diag(D) - W;
        % Tune the generalized eigenproblem
        A = (mu*L + (1-mu)*Ls); % (n1+n2) x (n1+n2) %  
%         A = mu*L + Ls; % (n1+n2) x (n1+n2) % 
        B = Ld;         % (n1+n2) x (n1+n2) %       
        % 3) Extract all features (now we can extract n dimensions!)
        KAK = K*A*K;
        KBK = K*B*K;
        [ALPHA LAMBDA] = gen_eig(KAK,KBK,'LM');        
        [LAMBDA j] = sort(diag(LAMBDA));
        ALPHA = ALPHA(:,j);        
        % 3b) check which projections must be inverted (with the 'mean of projected
        % samples per class' trick) and flip the axis that must be flipped
        E1 = ALPHA(1:n1,:);
        E2 = ALPHA(n1+1:end,:);
        sourceXpInv = (E1'*K1*-1)';
        sourceXp = (E1'*K1)';
        targetXp = (E2'*K2)';
        sourceXpInv = zscore(sourceXpInv);
        sourceXp = zscore(sourceXp);
        targetXp = zscore(targetXp);
  
        ErrRec = zeros(numel(unique(Y1)),size(ALPHA,2));
        ErrRecInv = zeros(numel(unique(Y1)),size(ALPHA,2));
        
        m1 = zeros(numel(unique(Y1)),size(ALPHA,2));
        m1inv = zeros(numel(unique(Y1)),size(ALPHA,2));
        m2 = zeros(numel(unique(Y1)),size(ALPHA,2));
        
        cls = unique(Y1);
        
        for j = 1:size(ALPHA,2)
            
            for i = 1:numel(unique(Y1))
                
                m1inv(i,j) = mean(sourceXpInv([Y1;Y1U]==cls(i),j));
                m1(i,j) = mean(sourceXp([Y1;Y1U]==cls(i),j));
                m2(i,j) = mean(targetXp([Y2;Y2U]==cls(i),j));
                
                ErrRec(i,j) = sqrt((mean(sourceXp([Y1;Y1U]==cls(i),j))-mean(targetXp([Y2;Y2U]==cls(i),j))).^2);
                ErrRecInv(i,j) = sqrt((mean(sourceXpInv([Y1;Y1U]==cls(i),j))-mean(targetXp([Y2;Y2U]==cls(i),j))).^2);
                
            end
        end
        
        
        mean(ErrRec);
        mean(ErrRecInv);
        
        Sc = max(ErrRec)>max(ErrRecInv);
        ALPHA(1:n1,Sc) = ALPHA(1:n1,Sc)*-1;
        
        % 4) Project the data
        nVectRBF = min(NF,rank(KBK));
        nVectRBF =  min(nVectRBF,rank(KAK));
        
%         for Nf = 1:nVectRBF
            
            E1     = ALPHA(1:n1,1:nVectRBF);
            E2     = ALPHA(n1+1:end,1:nVectRBF);
            Phi1toF = E1'*K1;
            Phi2toF = E2'*K2;
            
            Phi1TtoF = E1'*KT1;
            Phi2TtoF = E2'*KT2;
            
            X1_F= Phi1toF;
            X2_F=Phi2toF;
            X1_TF=Phi1TtoF ;
            X2_TF= Phi2TtoF;
           
            % 5) IMPORTAT: Normalize!!!!
            m1 = mean(Phi1toF');
            m2 = mean(Phi2toF');
            s1 = std(Phi1toF');
            s2 = std(Phi2toF');
            
            Phi1toF = zscore(Phi1toF')';
            Phi2toF = zscore(Phi2toF')';          
            Phi1TtoF = ((Phi1TtoF' - repmat(m1,2*t1,1))./ repmat(s1,2*t1,1))';
            Phi2TtoF = ((Phi2TtoF' - repmat(m2,2*t2,1))./ repmat(s2,2*t2,1))';
            
            % 6) Predict
% %             Ypred           = classify([Phi1toF(:,1:ncl*N)]',[Phi1toF(:,1:ncl*N),Phi2toF(:,1:ncl*N)]',[Y1;Y2]);
% %             Reslatent1Kernel2 = assessment(Y1,Ypred,'class');
%             
%             Ypred           = classify([Phi1TtoF]',[Phi1toF(:,1:ncl*N_source),Phi2toF(:,1:ncl*N_target)]',[Y1;Y2]);
%             Reslatent1Kernel2T = assessment(YT1,Ypred,'class');
%             
% %             Ypred           = classify([Phi2toF(:,1:ncl*N)]',[Phi1toF(:,1:ncl*N),Phi2toF(:,1:ncl*N)]',[Y1;Y2]);
% %             Reslatent2Kernel2 = assessment(Y2,Ypred,'class');
%             
%             Ypred           = classify([Phi2TtoF]',[Phi1toF(:,1:ncl*N_source),Phi2toF(:,1:ncl*N_target)]',[Y1;Y2]);
%             Reslatent2Kernel2T = assessment(YT2,Ypred,'class');
% %             
%             
% %             r1 = [r1; Reslatent1Kernel2.OA];
%             rT1 = [rT1; Reslatent1Kernel2T.OA];
%             
% %             r2 = [r2; Reslatent2Kernel2.OA];
%             rT2 = [rT2; Reslatent2Kernel2T.OA];
            
%         end
%         results.X1 = r1;
%         results.XT1 = rT1;
% %         results.X2 = r2;
%         results.XT2 = rT2;
        
%         [Best,Best_Dimension]=max(rT1);
%         [Best2,Best_Dimension2]=max(rT2);
%         
%             E1     = ALPHA(1:n1,1:Best_Dimension2);
%             E2     = ALPHA(n1+1:end,1:Best_Dimension2);
%             Phi1toF = E1'*K1;
%             Phi2toF = E2'*K2;
%             
%             Phi1TtoF = E1'*KT1;
%             Phi2TtoF = E2'*KT2;
%             
%             X1_F= Phi1toF;
%             X2_F=Phi2toF;
%             X1_TF=Phi1TtoF ;
%             X2_TF= Phi2TtoF;
%            
%             % 5) IMPORTAT: Normalize!!!!
%             m1 = mean(Phi1toF');
%             m2 = mean(Phi2toF');
%             s1 = std(Phi1toF');
%             s2 = std(Phi2toF');
%             
%             Phi1toF = zscore(Phi1toF')';
%             Phi2toF = zscore(Phi2toF')';          
%             Phi1TtoF = ((Phi1TtoF' - repmat(m1,2*t1,1))./ repmat(s1,2*t1,1))';
%             Phi2TtoF = ((Phi2TtoF' - repmat(m2,2*t2,1))./ repmat(s2,2*t2,1))';
%  figure;
% subplot(1,2,1)
%  scatter3(XT1(1,:),XT1(2,:),XT1(3,:),40,YT1,'o'), xlabel('Dim 1'),ylabel('Dim 2'),zlabel('Dim 3'),hold on, scatter3(XT2(1,:),XT2(2,:),XT2(3,:),40,YT2,'x'), xlabel('Dim 1'),ylabel('Dim 2'),zlabel('Dim 3'),colormap(jet)
%  title('original data classes (colors are classes)')
%  grid on 
%  axis image
% subplot(1,2,2)
%  h1=plot3(XT1(1,:),XT1(2,:),XT1(3,:),'ro');  xlabel('Dim 1'),ylabel('Dim 2'),zlabel('Dim 3'),hold on;h2=plot3(XT2(1,:),XT2(2,:),XT2(3,:),'x'); xlabel('Dim 1'),ylabel('Dim 2'),zlabel('Dim 3'),colormap(jet);
%  grid on
%  title('original data domains')
%  axis image
%  figure;
%  subplot(1,2,1)
%  scatter3(Phi1TtoF(1,:),Phi1TtoF(2,:),Phi1TtoF(3,:),40,YT1,'o'), xlabel('Dim 1'),ylabel('Dim 2'),zlabel('Dim 3') ,hold on, scatter3(Phi2TtoF(1,:),Phi2TtoF(2,:),Phi2TtoF(3,:),40,YT2,'x'), xlabel('Dim 1'),ylabel('Dim 2'),zlabel('Dim 3'),colormap(jet)
%  title('Aligned data classes (colors are classes)'),grid on
%   axis image
%  axis([-2.5 2.5 -2.5 2.5])
%  subplot(1,2,2)
%  h3=plot3(Phi1TtoF(1,:),Phi1TtoF(2,:),Phi1TtoF(3,:),'ro'), xlabel('Dim 1'),ylabel('Dim 2'),zlabel('Dim 3'); hold on; h4=plot3(Phi2TtoF(1,:),Phi2TtoF(2,:),Phi2TtoF(3,:),'x'), xlabel('Dim 1'),ylabel('Dim 2'),zlabel('Dim 3');colormap(jet);
%  title('Aligned data domains'),grid on
%   axis image
%  axis([-2.5 2.5 -2.5 2.5])    
       
end