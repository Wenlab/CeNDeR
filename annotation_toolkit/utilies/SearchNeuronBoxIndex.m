function NewBoxes=SearchNeuronBoxIndex(NewIdx,OldIdx,Boxes,current_volume,current_slice,manner)
    cv=current_volume;
    cs=current_slice;
    switch manner
    case "all the same"
        for s=1:width(Boxes)
            for n=1:length(Boxes{cv,s})
                if Boxes{cv,s}(n).idx==OldIdx
                    Boxes{cv,s}(n).idx=NewIdx;
                end
            end
        end
    case "this one and below"
        for s=cs:width(Boxes)
            for n=1:length(Boxes{cv,s})
                if Boxes{cv,s}(n).idx==OldIdx
                    Boxes{cv,s}(n).idx=NewIdx;
                end
            end
        end
    case "this one and above"
        for s=1:cs
            for n=1:length(Boxes{cv,s})
                if Boxes{cv,s}(n).idx==OldIdx
                    Boxes{cv,s}(n).idx=NewIdx;
                end
            end
        end
    otherwise
    end
    NewBoxes=Boxes;
end

        
