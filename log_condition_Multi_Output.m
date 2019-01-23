function Log_Condition = log_condition_Multi_Output(n,t,m,SI,dim_of_output,Tmu,Leaf_node_index,nonLeaf_node_index,TREE,alpha,beta,I)

         ni=[];P3_temp=[]; P4_temp2=[]; P4_temp1=[]; P5_temp=[];  R_bar=[]; 
         for i=1:length(Leaf_node_index{m,1})
             ni(Leaf_node_index{m,1}(i))   = length(TREE{m}(Leaf_node_index{m,1}(i)).NodeIndxes);                                                                                                                          
         end
         b=length(Leaf_node_index{m,1});
         k=length(nonLeaf_node_index{m,1});
         
         %----------- Log-Posterior ------------%
         P1 = +n/2*logdet(SI{1,t-1}); 
             
         P2 = +dim_of_output*b/2*log(Tmu);
                
         for i=1:b                  
             P3_temp(i) = -0.5*logdet(  ni(Leaf_node_index{m,1}(i)).*SI{1,t-1}  + Tmu.*I  );  %P3_temp(i) = -0.5* log(det(   ni(Leaf_node_index{m,1}(i))*SI{1,t-1}  + Tmu*I    ));
         end
         P3 = sum(P3_temp);
               
         for ii=1:b
             P4_temp1 = [];
             for i=1:ni(Leaf_node_index{m,1}(ii))
                 P4_temp1(i) = -0.5*(  TREE{m}(Leaf_node_index{m,1}(ii)).R(:,i)'   *  SI{1,t-1} *  TREE{m}(Leaf_node_index{m,1}(ii)).R(:,i)   );                           
             end
             P4_temp2(ii) = sum(P4_temp1);                               
         end
         P4 = sum(P4_temp2);               
              
         for ii=1:b
             for i=1:dim_of_output
                 R_bar(i,ii) = sum(TREE{m}(Leaf_node_index{m,1}(ii)).R(i,:))/ni(Leaf_node_index{m,1}(ii));
             end
             P5_temp(ii) = +0.5.* (ni(Leaf_node_index{m,1}(ii) )^2)  .*   (SI{1,t-1}*R_bar(:,ii))'  *   inv(ni(Leaf_node_index{m,1}(ii)).*SI{1,t-1}+Tmu.*I)    *  (SI{1,t-1}*R_bar(:,ii)   );                              
         end
         P5 = sum(P5_temp);
              
         Log_Posterior_BART = P1+P2+P3+P4+P5; 
                                                                
        %------------- Log-Prior --------------%
        Log_Prior_BART_temp1 =[];Log_Prior_BART_temp2=[];depth_term_node=[];depth_nonterm_node=[];H=[];
        H=cell2mat({TREE{m}(:).Children});
        for i=1:b                                     
             [~, kk]=find(H==Leaf_node_index{m,1}(i));                   
              Ancestors=[]; 
              Ancestors(1) = nonLeaf_node_index{m,1}(kk);
              while kk~=1     
                    [~, kk]=find(H==Ancestors(end));
                    Ancestors=[Ancestors kk];
              end
              depth_term_node(i) = length(Ancestors);
              Log_Prior_BART_temp1(i) = log(1-alpha*((1+depth_term_node(i))^beta));
         end
         for i=1:k
             if  nonLeaf_node_index{m,1}(i) ~=1
                 [~, kk]=find(H==nonLeaf_node_index{m,1}(i));                   
                 Ancestors=[]; 
                 Ancestors(1) = nonLeaf_node_index{m,1}(kk);
                 while kk~=1     
                       [~, kk]=find(H==Ancestors(end));
                       Ancestors=[Ancestors kk];
                 end
                 depth_nonterm_node(i) = length(Ancestors);
                 Log_Prior_BART_temp2(i) = log(alpha)+beta*log(1+depth_nonterm_node(i));
             else
                 Log_Prior_BART_temp2(i) = log(1-alpha);
             end
         end
         Log_Prior_BART = sum(Log_Prior_BART_temp1) + sum(Log_Prior_BART_temp2); 
                                                             
        %----------- Log-Condition ------------%
         Log_Condition  = Log_Posterior_BART + Log_Prior_BART;                          
        
end

