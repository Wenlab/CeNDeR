function [xlim,ylim]=zoom_shortcut(key,current_xlim,current_ylim,limit)
    x1=current_xlim(1);
    x2=current_xlim(2);
    xd=x2-x1;
    y1=current_ylim(1);
    y2=current_ylim(2);
    yd=y2-y1;
    switch key
    case {'equal'} % '+' or '=' is pressed, zoom in 2X
        xlim=[x1+xd/4,x2-xd/4];
        ylim=[y1+yd/4,y2-yd/4];
    case 'hyphen' % '-' is pressed zoom out 2X
        xlim=[x1-xd/2,x2+xd/2];
        ylim=[y1-yd/2,y2+yd/2];
        if xlim(1)<limit(1)
            xlim(1)=limit(1);
        end
        if ylim(1)<limit(1)
            ylim(1)=limit(1);
        end
        if xlim(2)>limit(2)
            xlim(2)=limit(2);
        end
        if ylim(2)>limit(2)
            ylim(2)=limit(2);
        end
    case 'backquote' % '`' is pressed restore view
        xlim=[limit(2)/2,limit(2)];
        ylim=[0,1025];
    case '1' % to a good single neuron view
        d=135;
        xbar=(limit(2)/2-d)/2;
        xlim=[limit(2)/2+xbar,limit(2)-xbar];
        ybar=(limit(2)-d)/2;
        ylim=[limit(1)+ybar,limit(2)-ybar];
        case '2' % to a middle size
            xd=256;
            yd=512;
            xbar=(limit(2)/2-xd)/2;
            xlim=[limit(2)/2+xbar,limit(2)-xbar];
            ybar=(limit(2)-yd)/2;
            ylim=[limit(1)+ybar,limit(2)-ybar];
    end
    
end

