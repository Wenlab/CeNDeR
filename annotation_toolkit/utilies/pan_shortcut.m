function [xlim,ylim]=pan_shortcut(key,current_xlim,current_ylim,limit)
    x1=current_xlim(1);
    x2=current_xlim(2);
    xd=x2-x1;
    y1=current_ylim(1);
    y2=current_ylim(2);
    yd=y2-y1;
    move_window_factor=4;
    switch key
    case {'leftarrow','j'} % pan to left
        xlim=current_xlim-xd/move_window_factor;
        ylim=current_ylim;
        if xlim(1)<limit(1)
            xlim=[limit(1),limit(1)+xd];
        end
    case {'rightarrow','l'} % pan to right
        xlim=current_xlim+xd/move_window_factor;
        ylim=current_ylim;
        if xlim(2)>limit(2)
            xlim=[limit(2)-xd,limit(2)];
        end
    case {'uparrow','i'} % pan up
        xlim=current_xlim;
        ylim=current_ylim-yd/move_window_factor;
        if ylim(1)<limit(1)
            ylim=[limit(1),limit(1)+yd];
        end
    case {'downarrow','k'} % pan down
        xlim=current_xlim;
        ylim=current_ylim+yd/move_window_factor;
        if ylim(2)>limit(2)
            ylim=[limit(2)-yd,limit(2)];
        end
    end
end