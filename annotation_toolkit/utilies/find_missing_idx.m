function missing_idx=find_missing_idx(neuron_idx)

N=max(neuron_idx);

for j=1:N
    
    if isempty(find(neuron_idx==j,1))
        missing_idx=j;
        return;
    end
end

missing_idx=N+1;
end
       
        
       