function [Child_left_targets, Child_right_targets, Child_left_indices, Child_right_indices, Terminal_indx_left, Terminal_indx_right ,Var_of_split, Loc_of_split,within_node_index_left,within_node_index_right,possible_to_split] = Node_Split2(Node_number, Node_Samples, Node_Targets, Node_indices ,min_pop,node_split_var,node_split_loc) 
 
[d,~] = size(Node_Samples);
possible_to_split = 0;
if  isnan(node_split_var) && isnan(node_split_loc)
    range = 1:1:d; F=[];
    for g=1:d          
        if length(unique(Node_Samples(g,:))) < 2
           F = [F g];
        end
    end
    if  ~isempty(F)
        for i=1:length(F)
            range=range(range~=F(i)) ;
        end     
    end
    if ~isempty(range)
       Var_of_split          = range(randi(length(range),1));
       possible_Loc_of_split = unique(Node_Samples(Var_of_split,:)); 
       possible_Loc_of_split = mean([possible_Loc_of_split(1:end-1);possible_Loc_of_split(2:end)]);
       possible_Loc_of_split = unique(possible_Loc_of_split);
       Loc_of_split          = possible_Loc_of_split(randi(length(possible_Loc_of_split)));
       possible_to_split     = 0;
    else
       possible_to_split     = 1;
       Var_of_split = NaN;
       Loc_of_split = NaN;
    end
else
    Var_of_split          = node_split_var;
    Loc_of_split          = node_split_loc;
    if  (length(unique(Node_Samples(Var_of_split,:)))<2) || (Loc_of_split >= max(Node_Samples(Var_of_split,:))) || (Loc_of_split <= min(Node_Samples(Var_of_split,:)))
        possible_to_split = 1;
    else
        possible_to_split = 0;
    end
end

if possible_to_split == 0
    
   Child_left_members        = Node_Samples(:,Node_Samples(Var_of_split,:)<=Loc_of_split);
   Child_right_members       = Node_Samples(:,Node_Samples(Var_of_split,:)> Loc_of_split);
   Child_left_targets        = Node_Targets(:,Node_Samples(Var_of_split,:)<=Loc_of_split);
   Child_right_targets       = Node_Targets(:,Node_Samples(Var_of_split,:)> Loc_of_split);
   Child_left_population     = size(Child_left_targets,2); 
   Child_right_population    = size(Child_right_targets,2);
   Child_left_indices_temp   = find(Node_Samples(Var_of_split,:)<=Loc_of_split);
   Child_right_indices_temp  = find(Node_Samples(Var_of_split,:)> Loc_of_split);
   within_node_index_left    = Child_left_indices_temp;
   within_node_index_right   = Child_right_indices_temp;
   Child_left_indices        = Node_indices(Child_left_indices_temp);
   Child_right_indices       = Node_indices(Child_right_indices_temp);     

   for g=1:d
       s(g) = length(unique(Child_left_members(g,:)));
       z(g) = length(unique(Child_right_members(g,:)));
   end
 
   if (length(within_node_index_left) <= min_pop) || all(s(:)<2)
       Terminal_indx_left = 1;
   else
       Terminal_indx_left = []; 
   end
 
   if (length(within_node_index_right)<= min_pop) || all(z(:)<2)
      Terminal_indx_right = 1; 
   else
      Terminal_indx_right = [];
   end
   
else
    Child_left_targets        =NaN;
    Child_right_targets       =NaN; 
    Child_left_population     =NaN; 
    Child_right_population    =NaN; 
    Child_left_indices_temp   =NaN; 
    Child_right_indices_temp  =NaN; 
    within_node_index_left    =NaN; 
    within_node_index_right   =NaN; 
    Child_left_indices        =NaN; 
    Child_right_indices       =NaN;    
    Terminal_indx_left        = []; 
    Terminal_indx_right       = [];
    return
    
end

end