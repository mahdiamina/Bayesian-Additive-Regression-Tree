 clc; close all; clear all;
% =================== Loading Dataset =================== %
Output       = load('Maire_Output.txt'); 
Input        = load('Maire_Input.txt');

% Input_Train  = load('Train_Input.txt');
% Output_Train = load('Train_Output.txt'); 
% 
% Input_Test   = load('Test_Input.txt'); 
% Output_Test  = load('Test_Output.txt'); 

Input_Train  = load('train_INPUT.txt');
Output_Train = load('train_OUTPUT.txt'); 

Input_Test   = load('test_INPUT.txt'); 
Output_Test  = load('test_OUTPUT.txt'); 


% ======================================================= %
Output_Column_To_Predict = 1;
Output_Train = Output_Train(:,Output_Column_To_Predict:5);
Output_Test  = Output_Test(:,Output_Column_To_Predict:5);
[mn,~]       = size(Input) ;
ix           = randperm(mn);

[~, dim_of_output]               = size(Output_Train);
[num_of_samples_Test,~]          = size(Input_Test);
[num_of_samples,Num_of_features] = size(Input_Train);

% =============== Standardisation of Data =============== %
Input_standard_Train  = (Input_Train-mean(Input_Train,1))./ std(Input_Train,1);
Output_standard_Train = (Output_Train-nanmean(Output_Train,1))./ nanstd(Output_Train,1);

Input_standard_Test   = (Input_Test-mean(Input_Train,1))./ std(Input_Train,1);
Output_standard_Test  = (Output_Test-nanmean(Output_Train,1))./ nanstd(Output_Train,1);

% ======================================================= %
Output_Classification_train = ones(size(Output_Train,1),size(Output_Train,2));
Output_Classification_test  = ones(size(Output_Test,1), size(Output_Test,2));

for i=1:dim_of_output
    
    NaNindex_Output_Train{i}   = find(isnan(Output_Train(:,i)));
    NaNindex_Output_Test{i}    = find(isnan(Output_Test(:,i)));
    nonNaNindex_Output_Test{i} = find(~isnan(Output_Test(:,i)));
    nonNaNindex_Output_Train{i}= find(~isnan(Output_Train(:,i)));
    
    Output_Classification_train(NaNindex_Output_Train{i},i) = 0;
    Output_Classification_test (NaNindex_Output_Test{i} ,i) = 0;
    
    Output_standard_Train(NaNindex_Output_Train{i},i) = 0;
    
end

%rng(123)
% ======================================================= %
n                       = num_of_samples;
train_patterns_M        = Input_standard_Train;
train_patterns_C        = train_patterns_M';

train_targets_M         = Output_standard_Train       ; train_targets_M = train_targets_M';
train_targets_C         = Output_Classification_train ; train_targets_C = train_targets_C';

test_targets_M          = Output_standard_Test';
test_targets_C          = Output_Classification_test';

original_index          = 1:1:num_of_samples;
num_of_trees_M          = 3;                 % <== Change Here
num_of_trees_C          = 6;                 % <== Change Here
num_of_iterat           = 320;               % <== Change Here 
min_popul               = 4;
v                       = 3;
Lambda                  = 0.1;
e                       = 2.0;
Tmu                     = 2;
alpha                   = 0.95;
beta                    = 2; beta = -beta;
ab_M                    = 1/(3*num_of_trees_M);
ab_C                    = 1/(3*num_of_trees_C);
z                       = zeros(1,num_of_samples);
tree_output_M           = cell(1,1);
Tree_Output_C           = cell(1,1);
I                       = eye(dim_of_output);
R_bar_M                 = []; 
R_bar_C                 = []; 
SI_M                    = cell(1,num_of_iterat);
SI_M{1,1}               = 1*eye(dim_of_output);
SI_C                    = cell(1,num_of_iterat);
SI_C{1,1}               = 1*eye(dim_of_output);
Sigma_M                 = cell(1,num_of_iterat);
Sigma_C                 = cell(1,num_of_iterat);
V                       = diag((1/dim_of_output)*ones(1,dim_of_output));
Prog_Ind                = waitbar(1/num_of_iterat); 
Tree_TempTest_Storage_M = zeros(dim_of_output,num_of_samples_Test,num_of_iterat);
Tree_TempTest_Storage_C = zeros(dim_of_output,num_of_samples_Test,num_of_iterat);

for m=1:num_of_trees_C
    TREE_Class{m}= struct('Children', cell(3,1), 'NodeIndxes',cell(3,1),'NodeTargets',cell(3,1),'R',cell(3,1),'NodeTerminalIndx',cell(3,1),'SplitVar', zeros(1,1), 'SplitLoc',zeros(1,1),'Mu',zeros(1,dim_of_output));
end

for ii=1:dim_of_output
    for i=1:num_of_samples
        if  train_targets_C(ii,i) == 1
            z(ii,i) = +3.5;    
        else
            z(ii,i) = -3.5;
        end
    end
end

for m=1:num_of_trees_C
    Num_of_nodes_C(m)            = 1;
    Num_of_teminal_node_C(m)     = 1;  
    TREE_Class{m}(1).Children    = [];              
    TREE_Class{m}(1).NodeIndxes  = original_index;
    TREE_Class{m}(1).NodeTargets = train_targets_C;  
    TREE_Class{m}(1).Z           = z;
end



for m=1:num_of_trees_C
    for i=1:dim_of_output
        Tree_Output_C{m}(i,:)= (repmat(TREE_Class{m}(1).Mu(i),1,n));  
    end
end
% ======================================================= %
for t=1:1                    
    for m=1:num_of_trees_C        
        % ============ R stump =========== %                                                       
        Sum_Of_All_Tree_Outputs_C = zeros(dim_of_output,n);
        for i=1:dim_of_output
            for mm=1:num_of_trees_C
                Sum_Of_All_Tree_Outputs_C(i,:) = Sum_Of_All_Tree_Outputs_C(i,:) + Tree_Output_C{mm}(i,:);        
            end
        end 
        for i=1:dim_of_output             
            TREE_Class{m}(1).R(i,:) = z(i,TREE_Class{m}(1).NodeIndxes) - (Sum_Of_All_Tree_Outputs_C(i,:)-Tree_Output_C{m}(i,:));
        end   
        
        % ========= Do the Action ======== %
        %           EMPTY at STUMP         %
        % ================================ %
                                                      
      % ========== Log-Posterior Stump ============ %                                 
        
        for ii=1:1
            for i=1:dim_of_output
                R_bar_C(i,ii) = sum(TREE_Class{m}(ii).R(i,:))/n;
            end
        end

        P1_C(t,m) = +n/2*logdet(SI_C{1,t});                  
        P2_C(t,m) = +dim_of_output*1/2*log(Tmu);                                  
        P3_C(t,m) = -0.5* logdet(n*SI_C{1,t}  + Tmu*I   );   
        for i=1:n
            P4_temp1(i) = -0.5* (   TREE_Class{m}(1).R(:,i)'   *  SI_C{1,t} *  TREE_Class{m}(1).R(:,i));                           
        end
        P4_C(t,m) = sum(P4_temp1);                               
        for i=1:dim_of_output
            R_bar_C(i,1) = sum(TREE_Class{m}(1).R(i,:))/n;
        end
        P5_C(t,m) = +0.5  *  n^2  *   (SI_C{1,t}*R_bar_C)'  *   inv(n*SI_C{1,t}+Tmu*I)   *  SI_C{1,t}*R_bar_C;  
        
        Log_Posterior_BART_C(m,t) = P1_C(t,m)+P2_C(t,m)+P3_C(t,m)+P4_C(t,m)+P5_C(t,m);                                                                                                                                         
        Log_Prior_BART_C(m,t)     = log(1-alpha);       
        Log_Condition_C(m,t)      = Log_Posterior_BART_C(m,t) + Log_Prior_BART_C(m,t);                                         
    % ============= Mu  & SI at stump ============== %         
        TREE_Class{m}(1).Mu = mvnrnd( inv(n*SI_C{1,t}+Tmu)*n*SI_C{1,t}*R_bar_C     ,   inv(n*SI_C{1,t}+Tmu)     ); 
    end
                   
    % ================ Trees Output ================ %
    for m=1:num_of_trees_C
        for i=1:dim_of_output
            Tree_Output_C{m}(i,:)= (repmat(TREE_Class{m}(1).Mu(i),1,n));  
        end
    end
    
    for  i=1:dim_of_output
         for mm=1:num_of_trees_C
             Sum_Of_All_Tree_Outputs_C(i,:) = Sum_Of_All_Tree_Outputs_C(i,:) + Tree_Output_C{mm}(i,:);        
         end
    end
                
    % =============== TESTING Class ================ %
    
    for  i=1:dim_of_output       
         Tree_TempTest_Storage_C(i,:,t) = repmat(TREE_Class{1}(1).Mu(i),1,num_of_samples_Test) ;         
    end
    
end  % << END OF ITERATION 1
 
for t=2:num_of_iterat    
    for m=1:num_of_trees_C 
    % -------------------------------- R -----------------------=-------- %
        ni=[];Tree_Temp1_C = zeros(dim_of_output,n);Tree_Temp2_C =zeros(dim_of_output,n); Sum_Of_All_Tree_Outputs_C =zeros(dim_of_output,n);
        Leaf_node_index_C{m,1} = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
        for  ii=1:dim_of_output
             for mm=1:num_of_trees_C
                 Tree_Temp1_C(ii,:) = Tree_Temp1_C(ii,:) + Tree_Output_C{mm}(ii,:);        
             end
        end
        for ii=1:dim_of_output
            Tree_Temp2_C(ii,:) = Tree_Temp1_C(ii,:) - Tree_Output_C{m}(ii,:);
        end
        for i=1:length(Leaf_node_index_C{m,1})
            ni(Leaf_node_index_C{m,1}(i)) = length(TREE_Class{m}(Leaf_node_index_C{m,1}(i)).NodeIndxes);                                                                                                                                  
            for ii=1:dim_of_output             
                TREE_Class{m}(Leaf_node_index_C{m,1}(i)).R(ii,:) = z(ii, TREE_Class{m}(Leaf_node_index_C{m,1}(i)).NodeIndxes) - (Tree_Temp2_C(ii,TREE_Class{m}(Leaf_node_index_C{m,1}(i)).NodeIndxes));                                     
            end          
        end         
            
    % ----- Random Number Generator to select Grow/Prune/Change/Swap ---- %
        if  t<5 || Num_of_nodes_C(m)<6
            rand_number = 1;
        else
            rand_number = randi(3,1);  
        end   
        switch rand_number                                                   
                                                  
%=========================================================================%
%======================%-------- Grow ----------%=========================%
%=========================================================================%
        case 1             
         % ------ Find a random possible node to do Grow ------ %                        
            possible_nodes_to_grow_temp1_C = [];possible_nodes_to_grow_temp2_C = [];
            possible_nodes_to_grow_temp1_C = Leaf_node_index_C{m,1};              
            possible_nodes_to_grow_temp2_C = find(~cellfun(@isempty,{TREE_Class{m}(1:length(TREE_Class{m})).NodeTerminalIndx}));
            if  isempty(possible_nodes_to_grow_temp2_C)
                possible_nodes_to_grow_C = possible_nodes_to_grow_temp1_C;       
            else
                SS =  compare2Arrays(possible_nodes_to_grow_temp1_C , possible_nodes_to_grow_temp2_C);
                possible_nodes_to_grow_temp1_C(SS) = [];
                possible_nodes_to_grow_C = possible_nodes_to_grow_temp1_C;                     
            end
            if  isempty(possible_nodes_to_grow_C)
                fprintf('Iter %d: Grow  Rejected  in tree <%d>, Due to NO more node exists with population above minimum.\n',t,m)            
                no_more_node_exist_to_split_C = 1;
                Acceptance_C(t,m) =0;
            else
                no_more_node_exist_to_split_C = 0;    
            end
            if no_more_node_exist_to_split_C == 0
               I_Grow_C = possible_nodes_to_grow_C (randi(length(possible_nodes_to_grow_C),1));
                             
         % ------------------ Do the Grow --------------------- %      
               PALM=[];PALM=TREE_Class{m};
               [TREE_Class{m}(Num_of_nodes_C(m)+1).NodeTargets , TREE_Class{m}(Num_of_nodes_C(m)+2).NodeTargets , TREE_Class{m}(Num_of_nodes_C(m)+1).NodeIndxes , TREE_Class{m}(Num_of_nodes_C(m)+2).NodeIndxes, TREE_Class{m}(Num_of_nodes_C(m)+1).NodeTerminalIndx, TREE_Class{m}(Num_of_nodes_C(m)+2).NodeTerminalIndx ,TREE_Class{m}(I_Grow_C).SplitVar, TREE_Class{m}(I_Grow_C).SplitLoc,within_node_index_left_C,within_node_index_right_C,possible_to_split_C]      =     Node_Split2(I_Grow_C, train_patterns_C(:,TREE_Class{m}(I_Grow_C).NodeIndxes), TREE_Class{m}(I_Grow_C).NodeTargets, TREE_Class{m}(I_Grow_C).NodeIndxes,min_popul,NaN,NaN);     
               TREE_Class{m}(I_Grow_C).Children =[Num_of_nodes_C(m)+1;Num_of_nodes_C(m)+2];    
               Num_of_nodes_C(m) = Num_of_nodes_C(m) + 2;
               TREE_Class{m}(Num_of_nodes_C(m)-1).R   =  TREE_Class{m}(I_Grow_C).R(:,within_node_index_left_C);
               TREE_Class{m}(Num_of_nodes_C(m)-0).R   =  TREE_Class{m}(I_Grow_C).R(:,within_node_index_right_C);
             
                 % ========= Log-Posterior Grow ======== %
               Leaf_node_index_C{m,1}    = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
               nonLeaf_node_index_C{m,1} = find(~cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));               
               Log_Condition_C(m,t)      = log_condition_Multi_Output(n,t,m,SI_C,dim_of_output,Tmu,Leaf_node_index_C,nonLeaf_node_index_C,TREE_Class,alpha,beta,I);%Log_Posterior_BART_M(m,t) + Log_Prior_BART_M(m,t);            
                
               
                    % ======= Check The Log-Condition of Grow ======= %            
               a = exp(Log_Condition_C(m,t) - Log_Condition_C(m,t-1));
               new_logpost_C=Log_Condition_C(m,t);   
               old_logpost_C=Log_Condition_C(m,t-1);  
               U = rand(1);
               if  t<6 || (a)>U
                   fprintf('Iter %d: Grow  ACCEPTED  in tree <%d>. New logpost = %.2f  |  Old logpost = %.2f \n',t,m,new_logpost_C,old_logpost_C) 
                   Acceptance_C(t,m) =1;                      
               else                    
                   fprintf('Iter %d: Grow  Rejected  in tree <%d>, Due to Lower Posterior Achieved. New logpost=%.2f  |  Old logpost=%.2f\n',t,m,new_logpost_C,old_logpost_C)                                                                                     
                   Num_of_nodes_C(m)= Num_of_nodes_C(m)-2; 
                   Acceptance_C(t,m)= 0;
                   TREE_Class{m}=[];TREE_Class{m} = PALM;                   
                   Leaf_node_index_C{m,1}    = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
                   nonLeaf_node_index_C{m,1} = find(~cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));                           
                   Log_Condition_C(m,t)      = log_condition_Multi_Output(n,t,m,SI_C,dim_of_output,Tmu,Leaf_node_index_C,nonLeaf_node_index_C,TREE_Class,alpha,beta,I);%Log_Posterior_BART_M(m,t) + Log_Prior_BART_M(m,t);                
               end
            end
            %=======================================================%
            %======================== PRUNE ========================%
            %=======================================================%
       case 2                  
             % ----- Find a random possible node to do Prune ------ %           
            Leaf_node_index_C{m,1} = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
            O_C = find(~ismember(1:1:Num_of_nodes_C(m),Leaf_node_index_C{m,1}));
            Node_Children = [];
            for aba = 1:Num_of_nodes_C(m)
                 if isempty(cell2mat({TREE_Class{m}(aba).Children}))
                    Node_Children(:,aba)= [0;0];
                 else              
                    Node_Children(:,aba)= cell2mat({TREE_Class{m}(aba).Children}); 
                 end
            end
            possible_nodes_to_prune_C = [];
            for f=1:length(O_C)
                 if all(Node_Children(:,Node_Children(1,O_C(f))) == 0) && all(Node_Children(:,Node_Children(2,O_C(f))) == 0)
                     possible_nodes_to_prune_C(f) = O_C(f);
                 end
            end  
            possible_nodes_to_prune_C = nonzeros(possible_nodes_to_prune_C);
            I_Prune_C = possible_nodes_to_prune_C(randi(length(possible_nodes_to_prune_C),1));                     
             
                          %----------- Do the Prune ---------------%
            PALM=[];PALM=TREE_Class{m};                                                  
            TREE_Class{m}(TREE_Class{m}(I_Prune_C).Children(2)) = [];
            TREE_Class{m}(TREE_Class{m}(I_Prune_C).Children(1)) = [];
             
            TREE_Class{m}(I_Prune_C).SplitVar = [];
            TREE_Class{m}(I_Prune_C).SplitLoc = [];
            Num_of_nodes_C(m)= Num_of_nodes_C(m)-2;
            for node_counter=1:length(TREE_Class{m})
                 if  ~cellfun(@isempty,{TREE_Class{m}(node_counter).Children}) && TREE_Class{m}(node_counter).Children(1) > TREE_Class{m}(I_Prune_C).Children(1)
                      TREE_Class{m}(node_counter).Children(1) = TREE_Class{m}(node_counter).Children(1) -2;
                      TREE_Class{m}(node_counter).Children(2) = TREE_Class{m}(node_counter).Children(2) -2;
                 end
            end
            TREE_Class{m}(I_Prune_C).Children = [];
                          
                       % ========= Log-Posterior Prune ======== %                            
             Leaf_node_index_C{m,1}    = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
             nonLeaf_node_index_C{m,1} = find(~cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));                        
             Log_Condition_C(m,t)      = log_condition_Multi_Output(n,t,m,SI_C,dim_of_output,Tmu,Leaf_node_index_C,nonLeaf_node_index_C,TREE_Class,alpha,beta,I);%Log_Posterior_BART_M(m,t) + Log_Prior_BART_M(m,t);               
            
                  % ======= Check The Log-Condition of Prune ======= %            
              a = exp(Log_Condition_C(m,t)-Log_Condition_C(m,t-1));
              new_logpost_C=Log_Condition_C(m,t); 
              old_logpost_C=Log_Condition_C(m,t-1);
              U = rand(1);
              if  a>1*U 
                  fprintf('Iter %d: Prune ACCEPTED  in tree <%d>. New logpost = %.2f  |  Old logpost = %.2f \n',t,m,new_logpost_C,old_logpost_C) 
                  Acceptance_C(t,m) = 1; 
              else                  
                  fprintf('Iter %d: Prune Rejected  in tree <%d>, Due to Lower Posterior Achieved. New logpost=%.2f  |  Old logpost=%.2f\n',t,m,new_logpost_C,old_logpost_C)                                                                                     
                  Num_of_nodes_C(m) = Num_of_nodes_C(m)+2; 
                  Acceptance_C(t,m) = 0;
                  TREE_Class{m}=[];TREE_Class{m} = PALM;                 
                  Leaf_node_index_C{m,1}    = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
                  nonLeaf_node_index_C{m,1} = find(~cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));                  
                  Log_Condition_C(m,t)      = log_condition_Multi_Output(n,t,m,SI_C,dim_of_output,Tmu,Leaf_node_index_C,nonLeaf_node_index_C,TREE_Class,alpha,beta,I);%Log_Posterior_BART_M(m,t) + Log_Prior_BART_M(m,t);  
             end
             
            %=======================================================%
            %======================== CHANGE =======================%
            %=======================================================%
            
       case 3                   
            % ------ Find a random possible node to do Change ----- %              
            Leaf_node_index_C{m,1} = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
            O_C = find(~ismember(1:1:Num_of_nodes_C(m),Leaf_node_index_C{m,1}));
            O_C(1)=[];possible_nodes_to_change_C = O_C;              
            I_Change_C = possible_nodes_to_change_C(randi(length(possible_nodes_to_change_C),1)); 
            %I_Change_C = max(possible_nodes_to_change_C);
            Node_Children = [];
            for aba = 1:Num_of_nodes_C(m)
                if isempty(cell2mat({TREE_Class{m}(aba).Children}))
                   Node_Children(:,aba) = [0;0];
                else              
                   Node_Children(:,aba) = cell2mat({TREE_Class{m}(aba).Children}); 
                end
            end
            S=0;FO=[];Grandchildren=[];  flaggg =1;              
            if Num_of_nodes_C(m) > I_Change_C
               Grandchildren = Node_Children(:,I_Change_C);                                
               if all(Grandchildren) ~=0
                  while flaggg
                        S=S+1; 
                        if S>length(Grandchildren)
                            flaggg =0;break ;
                        end
                        if Grandchildren(S)~=0 && flaggg ==1
                           FO=Node_Children(:,Grandchildren(S));                     
                           Grandchildren=[Grandchildren;  FO];                                         
                        end                     
                   end
                   Grandchildren = nonzeros(Grandchildren); 
                end                
            end
            
            ZZ_P=[I_Change_C; intersect(O_C,Grandchildren)];
            ZZ_C=[];ZZ_SV=[];ZZ_SL=[];
            for k=1:length(ZZ_P)
                ZZ_C(1:2,k)= Node_Children(:,ZZ_P(k));
            end 
            
                       % ========== Do the Change ============== %
            PALM=[];PALM=TREE_Class{m};  
            for k=1:length(ZZ_P)             
                [TREE_Class{m}(ZZ_C(1,k)).NodeTargets , TREE_Class{m}(ZZ_C(2,k)).NodeTargets , TREE_Class{m}(ZZ_C(1,k)).NodeIndxes , TREE_Class{m}(ZZ_C(2,k)).NodeIndxes, TREE_Class{m}(ZZ_C(1,k)).NodeTerminalIndx, TREE_Class{m}(ZZ_C(2,k)).NodeTerminalIndx ,TREE_Class{m}(ZZ_P(k)).SplitVar, TREE_Class{m}(ZZ_P(k)).SplitLoc,within_node_index_left_C,within_node_index_right_C,possible_to_split_C]      =     Node_Split2(ZZ_P(k), train_patterns_C(:,TREE_Class{m}(ZZ_P(k)).NodeIndxes), TREE_Class{m}(ZZ_P(k)).NodeTargets, TREE_Class{m}(ZZ_P(k)).NodeIndxes ,min_popul,NaN,NaN);                
                if possible_to_split_C==0
                   TREE_Class{m}(ZZ_C(1,k)).R   =  TREE_Class{m}(ZZ_P(k)).R(:,within_node_index_left_C);
                   TREE_Class{m}(ZZ_C(2,k)).R   =  TREE_Class{m}(ZZ_P(k)).R(:,within_node_index_right_C);
                else
                   TREE_Class{m} = PALM;  
                   break;    
                end
            end    
             
                       % ======== Log-Posterior Change ========= %
            Leaf_node_index_C{m,1} = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
            nonLeaf_node_index_C{m,1} = find(~cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));            
            Log_Condition_C(m,t)  =  log_condition_Multi_Output(n,t,m,SI_C,dim_of_output,Tmu,Leaf_node_index_C,nonLeaf_node_index_C,TREE_Class,alpha,beta,I);        
                      
                % ======= Check The Log-Condition of Change ======= %            
             a = exp(Log_Condition_C(m,t)-Log_Condition_C(m,t-1));
             new_logpost_C = Log_Condition_C(m,t);
             old_logpost_C = Log_Condition_C(m,t-1);
             U = rand(1);
             if  a>U && possible_to_split_C==0
                 fprintf('Iter %d: Chng  ACCEPTED  in tree <%d>. New logpost = %.2f  |  Old logpost = %.2f\n',t,m,new_logpost_C,old_logpost_C) 
                 Acceptance_C(t,m) = 1;                 
             else                
                 fprintf('Iter %d: Chng  Rejected  in tree <%d>, Due to Lower Posterior Achieved. New logpost=%.2f  |  Old logpost=%.2f\n',t,m,new_logpost_C,old_logpost_C)                                                                                                       
                 Acceptance_C(t,m) = 0;
                 TREE_Class{m}=[];TREE_Class{m} = PALM;
                 Leaf_node_index_C{m,1}    = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
                 nonLeaf_node_index_C{m,1} = find(~cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));                 
                 Log_Condition_C(m,t)      = log_condition_Multi_Output(n,t,m,SI_C,dim_of_output,Tmu,Leaf_node_index_C,nonLeaf_node_index_C,TREE_Class,alpha,beta,I);  
             end                     
              
             %=======================================================%
             %======================= SWAP ==========================%
             %=======================================================%                        
             
       case 4             
        % --- Find TWO random possible nodes (child & parent) to do Swap--%
             Swap_With_Prune_Happened=0;
             Leaf_node_index_C{m,1} = find(cellfun(@isempty,{TREE_Class{m}(1:length(TREE_Class{m})).Children}));
             O_C = find(~ismember(1:1:Num_of_nodes_C(m),Leaf_node_index_C{m,1}));
             Node_Children = [];
             for aba = 1:length(TREE_Class{m})
                 if isempty(cell2mat({TREE_Class{m}(aba).Children}))
                    Node_Children(:,aba)= [0;0];
                 else              
                    Node_Children(:,aba)= cell2mat({TREE_Class{m}(aba).Children}); 
                 end
             end
             possible_nodes_to_swap_C = [];
             for f=1:length(O_C)
                 if all(Node_Children(:,Node_Children(1,O_C(f))) == 0) && all(Node_Children(:,Node_Children(2,O_C(f))) == 0)
                     possible_nodes_to_swap_C(f) = O_C(f);
                 end
             end  
             possible_nodes_to_swap_C = nonzeros(possible_nodes_to_swap_C);
            %I_Swap_Child = possible_nodes_to_swap(randi(length(possible_nodes_to_swap),1)); 
             I_Swap_Child_C = max(possible_nodes_to_swap_C);
             I_Swap_Paren_C = [];
             for k=1:length(TREE_Class{m}) 
                 if any(TREE_Class{m}(k,:).Children==I_Swap_Child_C)==1
                    I_Swap_Paren_C = k;break
                 end
             end
             FE=4;grandchildren=[];              
             grandchildren = [grandchildren, Node_Children(:,I_Swap_Paren_C)];
             j=1;
             while j<=FE
                  if grandchildren(j)~=0
                     grandchildren = [grandchildren,  Node_Children(:,grandchildren(j))];  
                  end
                  j=j+1;
                  FE=numel(grandchildren);
              end
             ZZ_P=[];ZZ_P=[I_Swap_Paren_C; intersect(O_C,grandchildren)]; ZZ_C=[];            
             for k=1:length(ZZ_P)
                 ZZ_C(1:2,k)= Node_Children(:,ZZ_P(k));
             end
        
                      % ============= Do the SWAP ============ %                      
             PALM=[];PALM=TREE_Class{m};  
             for k=1:length(ZZ_P)                                                                                 
                 if ZZ_P(k) == I_Swap_Paren_C
                    [TREE_Class{m}(ZZ_C(1,k)).NodeTargets , TREE_Class{m}(ZZ_C(2,k)).NodeTargets , TREE_Class{m}(ZZ_C(1,k)).NodeIndxes , TREE_Class{m}(ZZ_C(2,k)).NodeIndxes, TREE_Class{m}(ZZ_C(1,k)).NodeTerminalIndx, TREE_Class{m}(ZZ_C(2,k)).NodeTerminalIndx ,TREE_Class{m}(ZZ_P(k)).SplitVar, TREE_Class{m}(ZZ_P(k)).SplitLoc,within_node_index_left_C,within_node_index_right_C,possible_to_split_C]      =     Node_Split2(ZZ_P(k), train_patterns_C(:,TREE_Class{m}(ZZ_P(k)).NodeIndxes), TREE_Class{m}(ZZ_P(k)).NodeTargets, TREE_Class{m}(ZZ_P(k)).NodeIndxes ,min_popul,PALM(I_Swap_Child_C).SplitVar,PALM(I_Swap_Child_C).SplitLoc);   %PALM(I_Swap_Child).SplitVar,PALM(I_Swap_Child).SplitLoc             
                    if possible_to_split_C==0
                       TREE_Class{m}(ZZ_C(1,k)).R  = TREE_Class{m}(ZZ_P(k)).R(:,within_node_index_left_C);
                       TREE_Class{m}(ZZ_C(2,k)).R  = TREE_Class{m}(ZZ_P(k)).R(:,within_node_index_right_C);                                            
                    else
                       break;    
                    end
                 end
                 if ZZ_P(k) == I_Swap_Child_C
                    [TREE_Class{m}(ZZ_C(1,k)).NodeTargets , TREE_Class{m}(ZZ_C(2,k)).NodeTargets , TREE_Class{m}(ZZ_C(1,k)).NodeIndxes , TREE_Class{m}(ZZ_C(2,k)).NodeIndxes, TREE_Class{m}(ZZ_C(1,k)).NodeTerminalIndx, TREE_Class{m}(ZZ_C(2,k)).NodeTerminalIndx ,TREE_Class{m}(ZZ_P(k)).SplitVar, TREE_Class{m}(ZZ_P(k)).SplitLoc,within_node_index_left_C,within_node_index_right_C,possible_to_split_C]      =     Node_Split2(ZZ_P(k), train_patterns_C(:,TREE_Class{m}(ZZ_P(k)).NodeIndxes), TREE_Class{m}(ZZ_P(k)).NodeTargets, TREE_Class{m}(ZZ_P(k)).NodeIndxes ,min_popul,PALM(I_Swap_Paren_C).SplitVar,PALM(I_Swap_Paren_C).SplitLoc);   %PALM(I_Swap_Child).SplitVar,PALM(I_Swap_Child).SplitLoc             
                    if possible_to_split_C==0
                       TREE_Class{m}(ZZ_C(1,k)).R  = TREE_Class{m}(ZZ_P(k)).R(:,within_node_index_left_C);
                       TREE_Class{m}(ZZ_C(2,k)).R  = TREE_Class{m}(ZZ_P(k)).R(:,within_node_index_right_C);                                            
                    else
                       break;    
                    end
                 end
                 if (ZZ_P(k) ~= I_Swap_Child_C) && (ZZ_P(k) ~= I_Swap_Paren_C)
                    [TREE_Class{m}(ZZ_C(1,k)).NodeTargets , TREE_Class{m}(ZZ_C(2,k)).NodeTargets , TREE_Class{m}(ZZ_C(1,k)).NodeIndxes , TREE_Class{m}(ZZ_C(2,k)).NodeIndxes, TREE_Class{m}(ZZ_C(1,k)).NodeTerminalIndx, TREE_Class{m}(ZZ_C(2,k)).NodeTerminalIndx ,TREE_Class{m}(ZZ_P(k)).SplitVar, TREE_Class{m}(ZZ_P(k)).SplitLoc,within_node_index_left_C,within_node_index_right_C,possible_to_split_C]      =     Node_Split2(ZZ_P(k), train_patterns_C(:,TREE_Class{m}(ZZ_P(k)).NodeIndxes), TREE_Class{m}(ZZ_P(k)).NodeTargets, TREE_Class{m}(ZZ_P(k)).NodeIndxes ,min_popul,NaN,NaN);   %PALM(I_Swap_Child).SplitVar,PALM(I_Swap_Child).SplitLoc             
                    if possible_to_split_C==0
                       TREE_Class{m}(ZZ_C(1,k)).R  = TREE_Class{m}(ZZ_P(k)).R(:,within_node_index_left_C);
                       TREE_Class{m}(ZZ_C(2,k)).R  = TREE_Class{m}(ZZ_P(k)).R(:,within_node_index_right_C);                                            
                    else
                       break;    
                    end
                 end
                 
            end
             
                     % ========= Log-Posterior Swap ======== %
            if possible_to_split_C==0  
               Leaf_node_index_C{m,1}    = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
               nonLeaf_node_index_C{m,1} = find(~cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));                              
               Log_Condition_C(m,t)      = log_condition_Multi_Output(n,t,m,SI_C,dim_of_output,Tmu,Leaf_node_index_C,nonLeaf_node_index_C,TREE_Class,alpha,beta,I);%Log_Posterior_BART_C(m,t); %+ Log_Prior_BART_C(m,t);  
            end
                   % ======= Check The Log-Condition of Swap ======= %            
            if  possible_to_split_C==0
                a = exp(Log_Condition_C(m,t)-Log_Condition_C(m,t-1));  
                new_logpost_C = Log_Condition_C(m,t); 
                old_logpost_C = Log_Condition_C(m,t-1);
                U = rand(1);
            end             
            if  a<U || possible_to_split_C==1
                if possible_to_split_C==0
                    fprintf('Iter %d: Swap  Rejected  in tree <%d>, Due to Lower Posterior Achieved. New logpost=%.2f  |  Old logpost=%.2f\n',t,m,new_logpost_C,old_logpost_C)                                                                                                       
                else
                    fprintf('Iter %d: Swap  Rejected  in tree <%d>, Due to UNMATCHED Swap Parameters.NO New logpost  |  Old logpost=%.2f \n',t,m,old_logpost_C)     
                end
                Acceptance_C(t,m) =0;
                TREE_Class{m}=[];TREE_Class{m} = PALM;                 
                Leaf_node_index_C{m,1}    = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
                nonLeaf_node_index_C{m,1} = find(~cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));                                
                Log_Condition_C(m,t)      = log_condition_Multi_Output(n,t,m,SI_C,dim_of_output,Tmu,Leaf_node_index_C,nonLeaf_node_index_C,TREE_Class,alpha,beta,I); %Log_Posterior_BART_C(m,t);% + Log_Prior_BART_C(m,t);                                    
            else
                fprintf('Iter %d: Swap  ACCEPTED  in tree <%d>. New logpost = %.2f  |  Old logpost = %.2f\n',t,m,new_logpost_C,old_logpost_C) 
                Acceptance_C(t,m) =1;           
            end                                                                                                          
            
         end %<< end of 'Case'
          
         % =============== Update Mu & SI ================= %    
         Leaf_node_index_C{m,1} = find(cellfun(@isempty,{TREE_Class{m}(1:Num_of_nodes_C(m)).Children}));
         ni=[]; R_bar_C=[];      
         for ii=1:length(Leaf_node_index_C{m,1})
             ni(Leaf_node_index_C{m,1}(ii)) = length(TREE_Class{m}(Leaf_node_index_C{m,1}(ii)).NodeIndxes); 
             for i=1:dim_of_output
                 R_bar_C(i,ii) = sum(TREE_Class{m}(Leaf_node_index_C{m,1}(ii)).R(i,:))/ni(Leaf_node_index_C{m,1}(ii));
             end                     
         end
         for i=1:length(Leaf_node_index_C{m,1})
             Mean_C  = []; SD_C=[];                                                                                                                                                                                                                                                
             Mean_C  = inv(ni(Leaf_node_index_C{m,1}(i))*SI_C{1,t-1}+Tmu)*ni(Leaf_node_index_C{m,1}(i))*SI_C{1,t-1}*R_bar_C(:,i);      %
             SD_C    = ((   inv(ni(Leaf_node_index_C{m,1}(i))*SI_C{1,t-1}+Tmu)   ));              
             TREE_Class{m}(Leaf_node_index_C{m,1}(i)).Mu = mvnrnd(Mean_C,SD_C);           
         end          
           
        % ============= Finding Tree Outputs =============== %
         Node_Children = [];Node_SplitVar=[];Node_SplitLoc=[];
         for aba = 1:Num_of_nodes_C(m)
             if isempty(cell2mat({TREE_Class{m}(aba).Children}))
                Node_Children(:,aba) = [0;0];
                Node_SplitVar(aba)   = 0;
                Node_SplitLoc(aba)   = 0;
             else              
                Node_Children(:,aba) = cell2mat({TREE_Class{m}(aba).Children}); 
                Node_SplitVar(aba)   = cell2mat({TREE_Class{m}(aba).SplitVar});
                Node_SplitLoc(aba)   = cell2mat({TREE_Class{m}(aba).SplitLoc});
             end
         end                                                                                                    
         train_samples = train_patterns_C';
         for nn=1:num_of_samples
             flag = 0; nodeindexx = 1;
             while flag == 0
                   if train_samples(nn,Node_SplitVar(nodeindexx))<=Node_SplitLoc(nodeindexx)
                      nodeindexx = Node_Children(1,nodeindexx);
                   else
                      nodeindexx = Node_Children(2,nodeindexx);
                   end                   
                   if all(Node_Children(:,nodeindexx) == 0)
                      flag = 1;
                      output_C(:,nn) = TREE_Class{m}(nodeindexx).Mu(:);%Node_MU(nodeindexx);
                   end
             end
          end
          Tree_Output_C{m} = output_C;
         %--------------------------------                                                      
    end   % end Num_of_trees 
                   
    % =================== Update z Values ===================== %        
    Sum_Of_All_Tree_Outputs_C = zeros(dim_of_output,n);
    for i=1:dim_of_output
        for mm=1:num_of_trees_C
            Sum_Of_All_Tree_Outputs_C(i,:) = Sum_Of_All_Tree_Outputs_C(i,:) + Tree_Output_C{mm}(i,:);        
        end
    end
    for ii=1:dim_of_output
        for i=1:num_of_samples
            if train_targets_C(ii,i) == 1
               z(ii,i) = max(normrnd(Sum_Of_All_Tree_Outputs_C(ii,i),1), 0);    
            else
               z(ii,i) = min(normrnd(Sum_Of_All_Tree_Outputs_C(ii,i),1), 0);
            end
        end
    end    
    SI_C{1,t} = 1*eye(dim_of_output);  
                
    % ================== TESTING Data Estimate ================ %
    
    tree_output_test_C=[];
    test_samples_C = Input_standard_Test;    
    for mm=1:num_of_trees_C
        Node_Children = [];Node_SplitVar=[];Node_SplitLoc=[];output_test=[];
        for aba = 1:Num_of_nodes_C(mm)
            if  isempty(cell2mat({TREE_Class{mm}(aba).Children}))
                Node_Children(:,aba) = [0;0];
                Node_SplitVar(aba)   = 0;
                Node_SplitLoc(aba)   = 0;
            else              
                Node_Children(:,aba) = cell2mat({TREE_Class{mm}(aba).Children}); 
                Node_SplitVar(aba)   = cell2mat({TREE_Class{mm}(aba).SplitVar});
                Node_SplitLoc(aba)   = cell2mat({TREE_Class{mm}(aba).SplitLoc});
            end
        end
        for nn=1:num_of_samples_Test
            flag = 0; nodeindexx = 1;
            while flag == 0
                  if test_samples_C(nn,Node_SplitVar(nodeindexx))<=Node_SplitLoc(nodeindexx)
                     nodeindexx = Node_Children(1,nodeindexx);
                  else
                     nodeindexx = Node_Children(2,nodeindexx);
                  end                   
                  if all(Node_Children(:,nodeindexx) == 0)
                     flag = 1;
                     output_test_C(:,nn) = TREE_Class{mm}(nodeindexx).Mu(:);
                  end
             end
        end
        tree_output_test_C{mm} = output_test_C;
    end       
    for i=1:dim_of_output
        for mm=1:num_of_trees_C
            Tree_TempTest_Storage_C(i,:,t) = Tree_TempTest_Storage_C(i,:,t) + tree_output_test_C{mm}(i,:);        
        end         
    end    
    disp('---------------------------------------------------')           
    waitbar(t/num_of_iterat,Prog_Ind,'Processing...');       
                 
end         % end of iteration 
close(Prog_Ind)
% ====================== TESTING  classification ======================= %

for i=1:dim_of_output
    Tree_test_C(i,:) = normcdf(mean(Tree_TempTest_Storage_C(i,:,min(round(num_of_iterat/2.0),999):end),3)); 
end   
   
% ====================== Plotting Classification ======================== %
subplotRowsColumns = numSubplots(num_of_trees_C);

                %------ Plot Log-Condition Curves -------%
figure(5)


set(gcf,'Position', get(gcf,'Position') + [0,0,150,0]);
pos2 = get(gcf,'Position'); 
set(gcf,'Position',  [755,450,390,350])
for m=1:num_of_trees_C    
    subplot(subplotRowsColumns(1),subplotRowsColumns(2),m);plot(Log_Condition_C(m,1:end),'-o','MarkerSize',5,'MarkerEdgeColor',[0.5 0 0.5],'MarkerFaceColor',[.5 .5 .9],'LineWidth',2);ylabel(['logPOSTERIOR Tree ' num2str(m) '']); grid on;hold on    
end  

%               %---------- Plot training AUC ----------%
                
figure(6) 
set(gcf,'Position', get(gcf,'Position') + [0,0,150,0]);
pos2 = get(gcf,'Position'); 
set(gcf,'Position',[755,30,390,350]) % Shift position of Figure(1)
%Color = {'b','g','r','k','y','c'};
[AUC,fpr,tpr] = AUC(logical(Output_Classification_test),Tree_test_C',1);grid on,hold on
legend('Out1','Out2','Out3','Out4','Out5');
xlabel('False Positive Rate');
ylabel('True Positive Rate');


                %------------ Plot Trees ----------------%
Node_SplitLoc_Tags = []; Node_Children = []; Node_SplitVar =[];Node_SplitLoc=[]; 
%Node_Mu(aba) = [];x=[];y=[];parent=[];Node_Number_Tags =[];

for m=1:num_of_trees_C 
    for aba=1:Num_of_nodes_C(m)    
        if isempty(cell2mat({TREE_Class{m}(aba).Children}))
           Node_Children(:,aba)= [0;0];
           Node_SplitVar(aba)  = 0;
           Node_SplitLoc(aba)  = 0;
%           Node_Mu(aba)        = cell2mat({normcdf(TREE_Class{m}(aba).Mu)});
        else              
           Node_Children(:,aba)= cell2mat({TREE_Class{m}(aba).Children}); 
           Node_SplitVar(aba)  = cell2mat({TREE_Class{m}(aba).SplitVar});
           Node_SplitLoc(aba)  = cell2mat({TREE_Class{m}(aba).SplitLoc});
%           Node_Mu(aba)        = zeros(1,dim_of_output);
        end
    end
    parent = zeros(1,Num_of_nodes_C(m));
    for i=2:Num_of_nodes_C(m)
        [~, kk]=find(Node_Children==i);
        parent(1,i) = kk;
    end
   figure(7)
   pos3 = get(gcf,'Position'); % get position of Figure(1) 
   
   set(gcf,'Position',[1145,450,390,350]) 
   subplot(subplotRowsColumns(1),subplotRowsColumns(2),m);treeplot(parent,'square','k');set(gca,'YTick',[],'XTick',[]);grid on
   [x,y] = treelayout(parent);x = x';y = y';
   Node_Number_Tags   = cellstr(num2str((1:size(parent,2))'));
   Node_SplitVar_Tags = cellstr(num2str((Node_SplitVar)'));
   Node_SplitLoc_Tags = cellstr(num2str((Node_SplitLoc)'));
  % Node_Mu_Tags       = cellstr(num2str((Node_Mu)'));
   text(x(1:end,1), y(1:end,1), Node_Number_Tags, 'VerticalAlignment','bottom','HorizontalAlignment','right','color','k','FontSize',11)
   for g=1:Num_of_nodes_C(m)   
       if Node_Children(1,g) ~= 0  
          text(x(g,1), y(g,1), Node_SplitVar_Tags{g}, 'VerticalAlignment','top','HorizontalAlignment','right','color','r','FontSize',8)
          text(x(g,1), y(g,1), Node_SplitLoc_Tags{g}, 'VerticalAlignment','bottom','color','b','FontSize',6)
       else
    %      text(x(g,1)-0.04, y(g,1)-0.05, Node_Mu_Tags{g}, 'VerticalAlignment','bottom','HorizontalAlignment','left', 'color',[0, 0.3, 0],'FontWeight','bold','FontSize',8)
       end
   end
   title(['Num of Nodes=' num2str(Num_of_nodes_C(m))],'FontSize',10,'FontName','Times New Roman');
end

%       ------------ Plot Testing Predicted Vs. Actual ----------------%
figure(8)
for i=1:dim_of_output
    MIN_C(i) = 0;
    MAX_C(i) = 1;
end
set(gcf,'Position', get(gcf,'Position') + [0,0,150,0]);
pos4 = get(gcf,'Position'); 
set(gcf,'Position',  [1145,30,390,345])
subplotRowsColumns = numSubplots(dim_of_output);
for i=1:dim_of_output
    subplot(subplotRowsColumns(1),subplotRowsColumns(2),i);plot(jitter(test_targets_C(i,:),0.7),Tree_test_C(i,:),'o','LineWidth',2);grid on;xlabel('Actual');ylabel('Predicted');hold on;axis tight; fplot(@(x) x,[MIN_C(i) MAX_C(i)],'m','LineWidth',2);axis tight
    Coeff_Test=corrcoef(test_targets_C(i,:),Tree_test_C(i,:));
    Coeff_Train=corrcoef(train_targets_C(i,:),normcdf(Sum_Of_All_Tree_Outputs_C(i,:)));
    title(['Testing Corr Coeff: ' num2str(Coeff_Test(1,2)) ' ----- Training Corr Coeff:' num2str(Coeff_Train(1,2))])
end

AUC

