function varargout = whole_brain_imaging(varargin)
    %several inputs:
    %1. image stack
    %2. neuronal_position
    %3. neuronal_idx
    %4. ROIposition

    % whole_brain_imaging MATLAB code for whole_brain_imaging.fig
    %      whole_brain_imaging, by itself, creates a new whole_brain_imaging or raises the existing
    %      singleton*.
    %
    %      H = whole_brain_imaging returns the handle to a new whole_brain_imaging or the handle to
    %      the existing singleton*.
    %
    %      whole_brain_imaging('CALLBACK',hObject,eventData,handles,...) calls the local
    %      function named CALLBACK in whole_brain_imaging.M with the given input arguments.
    %
    %      whole_brain_imaging('Property','Value,...) creates a new whole_brain_imaging or raises the
    %      existing singleton*.  Starting from the left, property value pairs are
    %      applied to the GUI before whole_brain_imaging_OpeningFcn gets called.  An
    %      unrecognized property name or invalid value makes property application
    %      stop.  All inputs are passed to whole_brain_imaging_OpeningFcn via varargin.
    %
    %      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
    %      instance to run (singleton)".
    %
    % See also: GUIDE, GUIDATA, GUIHANDLES

    % Edit the above text to modify the response to help whole_brain_imaging

    % Last Modified by GUIDE v2.5 13-Jan-2022 16:11:58

    % Begin initialization code - DO NOT EDIT
    gui_Singleton = 1;
    gui_State = struct('gui_Name', mfilename, ...
        'gui_Singleton', gui_Singleton, ...
        'gui_OpeningFcn', @whole_brain_imaging_OpeningFcn, ...
        'gui_OutputFcn', @whole_brain_imaging_OutputFcn, ...
        'gui_LayoutFcn', [], ...
        'gui_Callback', []);

    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end

    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end

end

% End initialization code - DO NOT EDIT

% --- Executes just before whole_brain_imaging is made visible.
function whole_brain_imaging_OpeningFcn(hObject, eventdata, handles, varargin)
    % This function has no output args, see OutputFcn.
    % hObject    handle to figure
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    % varargin   command line arguments to whole_brain_imaging (see VARARGIN)

    % Choose default command line output for whole_brain_imaging

    handles.img_stack = varargin{1};
    handles.volumes = length(handles.img_stack);
    %img_stack{1,1} is the first volume
    [handles.image_height, handles.image_width, handles.slices] = size(handles.img_stack{1, 1});
    screensize = get(0, 'ScreenSize');
    width = 800;
    height = screensize(4) - 230;

    set(hObject, 'Units', 'pixels');
    % set position of the GUI window
    set(handles.figure1, 'Position', [100 400 width height + 80]);
    set(handles.slider1, 'Position', [0 0 width 18]);
    set(handles.slider2, 'Position', [0 18 width 18]);
    axes(handles.axes1);
    handles.low = 100;
    handles.high = 350;
    handles.tracking_threshold = 70;
    handles.contrast_enhancement = 0;
    %img_stack{1,1}(:,:,1) is the first volume, and its first slice
    if handles.contrast_enhancement
        img = imagesc(contrast_enhancement(handles.img_stack{1, 1}(:, :, 1)), [handles.low handles.high]);
    else
        img = imagesc(handles.img_stack{1, 1}(:, :, 1), [handles.low handles.high]);
    end

    colormap(gray);
    set(handles.axes1, ...
        'Visible', 'off', ...
        'Units', 'pixels', ...
        'Position', [0 30 width height]);

    handles.axel_width = width;
    handles.axel_height = height;
    % set up volume number and slice number display
    set(handles.text1, 'Units', 'pixels');
    set(handles.text1, 'Position', [0 44 + height + 2 width 18]);
    set(handles.text1, 'HorizontalAlignment', 'center');
    set(handles.text1, 'String', strcat(num2str(1), '/', num2str(handles.volumes), ' volumes; ', num2str(1), '/', num2str(handles.slices), ' z slices'));
    % set up image pixel info display
    handles.hp = impixelinfo;
    set(handles.hp, 'Position', [0 44 + height + 2, 300, 20]);
    %set up find box edit display
    set(handles.edit2,'Units','pixels');
    set(handles.findNeuronText,'Units','pixels');
    set(handles.neuronText,'Units','pixels');
    set(handles.edit2,'Position',[width-50 44+height+15 50 20]);
    set(handles.findNeuronText,'Position',[width-200 44+height+12 150 20]);
    set(handles.neuronText,'Position',[width-200 35+height 200 25]);
    set(handles.neuronText,'String','');
    set(handles.findNeuronText,'String','Find neuron in this volume:');

    set(img, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDown_Callback'',gcbo,[],guidata(gcbo))');
    
    %the maxiumum number of neurons to track is 302
    handles.points = cell(302, 1);
    handles.colorset = varycolor(302);
    handles.lr_tform = [];
    %initialize neuronal position;
    handles.calcium_signals = cell(handles.volumes, 1);
    % initialize Box
    Box = make_a_BOX();

    switch length(varargin)

        case 1

            handles.neuronal_position = cell(handles.volumes, 2);
            handles.neuronal_idx = cell(handles.volumes, 1);
            handles.Boxes = repmat({Box}, handles.volumes, handles.slices);
            handles.ROIposition = cell(handles.volumes, handles.slices);

            for i = 1:handles.volumes

                for j = 1:handles.slices
                    handles.ROIposition{i, j}(1, 1) = 1;
                    handles.ROIposition{i, j}(1, 2) = 1;
                    handles.ROIposition{i, j}(1, 3) = handles.image_width - 1;
                    handles.ROIposition{i, j}(1, 4) = handles.image_height - 1;
                end

            end

        case 2
            handles.neuronal_position = cell(handles.volumes, 2);
            handles.neuronal_idx = cell(handles.volumes, 1);
            handles.Boxes = varargin{2};
            handles.ROIposition = cell(handles.volumes, handles.slices);

            for i = 1:handles.volumes

                for j = 1:handles.slices
                    handles.ROIposition{i, j}(1, 1) = 1;
                    handles.ROIposition{i, j}(1, 2) = 1;
                    handles.ROIposition{i, j}(1, 3) = handles.image_width - 1;
                    handles.ROIposition{i, j}(1, 4) = handles.image_height - 1;
                end

            end

        case 3

            handles.neuronal_position = varargin{2};
            handles.neuronal_idx = varargin{3};
            N = max(handles.neuronal_idx{1, 1});
            handles.colorset = varycolor(N);
            handles.Boxes = repmat({Box}, handles.volumes, handles.slices);
            handles.ROIposition = cell(handles.volumes, handles.slices);

            for i = 1:handles.volumes

                for j = 1:handles.slices
                    handles.ROIposition{i, j}(1, 1) = 1;
                    handles.ROIposition{i, j}(1, 2) = 1;
                    handles.ROIposition{i, j}(1, 3) = handles.image_width - 1;
                    handles.ROIposition{i, j}(1, 4) = handles.image_height - 1;
                end

            end

        case 4

            handles.neuronal_position = varargin{2};
            handles.neuronal_idx = varargin{3};
            handles.Boxes = varargin{4};
            handles.ROIposition = cell(handles.volumes, handles.slices);
            N = max(handles.neuronal_idx{1, 1});
            handles.colorset = varycolor(N);
        case 5
            handles.neuronal_position = varargin{2};
            handles.neuronal_idx = varargin{3};
            handles.Boxes = varargin{4};
            handles.ROIposition = varargin{5};
            N = max(handles.neuronal_idx{1, 1});
            handles.colorset = varycolor(N);
        otherwise

            disp('Error: exceeds the number of inputs');

    end

    handles.rg_tform = [];

    handles.slider1_is_active = 0;
    handles.slider2_is_active = 1;

    handles.signal = [];
    handles.normalized_signal = [];
    handles.ratio = [];

    min_step = 1 / (handles.volumes - 1);
    max_step = 5 * min_step;
    set(handles.slider1, ...
        'Enable', 'on', ...
        'Min', 1, ...
        'Max', handles.volumes, ...
        'Value', 1, ...
        'SliderStep', [min_step max_step]);

    min_step = 1 / (handles.slices - 1);
    max_step = 5 * min_step;
    set(handles.slider2, ...
        'Enable', 'on', ...
        'Min', 1, ...
        'Max', handles.slices, ...
        'Value', 1, ...
        'SliderStep', [min_step max_step]);

    handles.previous_volume = 1;
    handles.previous_slice = 1;
    handles.current_volume = 1;
    handles.current_slice = 1;
    handles.reference_volume = 1;
    handles.fontsize = 10;

    %ROI

    if ~isempty(handles.ROIposition{handles.current_volume, handles.current_slice})
        rect = handles.ROIposition{handles.current_volume, handles.current_slice};
        handles.ROI{handles.current_volume, handles.current_slice} = rectangle('Curvature', [0 0], 'Position', rect, 'EdgeColor', 'y');
        set(handles.ROI{handles.current_volume, handles.current_slice}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownROI_Callback'',gcbo,[],guidata(gcbo))');

    elseif ~isempty(handles.ROIposition{max(handles.current_volume - 1, 1), handles.current_slice})
        rect = handles.ROIposition{max(handles.current_volume - 1, 1), handles.current_slice};
        handles.ROI{handles.current_volume, handles.current_slice} = rectangle('Curvature', [0 0], 'Position', rect, 'EdgeColor', 'y');
        handles.ROIposition{handles.current_volume, handles.current_slice} = rect;
        set(handles.ROI{handles.current_volume, handles.current_slice}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownROI_Callback'',gcbo,[],guidata(gcbo))');
    end

    handles.ButtonDownFunc = 'point';
    handles.output = hObject;
    handles.refreshed_times = 0;
    handles.bounding_neuron=0;
    handles.DispBox = 0; % default not to display boxes
    handles.DispBoxID = 1;
    handles.Display_BOX_ID.Checked='on';
    handles.ShowMeanVal=0;
    handles.boxLocation=[];
    handles.neuron2find=[];
    handles.hBoxes = images.roi.Rectangle.empty;
    handles.box_nxt_id = 0;
    handles.box_id_before = 0;
    handles.Assign_index_to_box = 0;
    handles.last_outlier_box_idx = 1;
    handles.last_outlier_box_choice = "only this one";
    handles.CmapJet=1;
    handles.colormap_jet.Checked='on';
    % logical retrive from or push to figure handle of a value:
    % 1 is retrieve from slider, 0 is push to slider
    handles.if_retrieve = 0;
    % re-sequence box ID
    n = 1;
    id = 0;

    for v = 1:size(handles.Boxes, 1)

        for s = 1:size(handles.Boxes, 2)

            if ~isempty(handles.Boxes{v, s})

                for b = 1:numel(handles.Boxes{v, s})
                    handles.Boxes{v, s}(b).identifier = n;
                    n = n + 1;
                end

            end

        end

    end

    handles.box_nxt_id = n;
    % Update handles structure
    guidata(hObject, handles);
end

% UIWAIT makes whole_brain_imaging wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = whole_brain_imaging_OutputFcn(hObject, eventdata, handles)
    % varargout  cell array for returning output args (see VARARGOUT);
    % hObject    handle to figure
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)

    % Get default command line output from handles structure
    varargout{1} = handles.output;
end

% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
    % hObject    handle to FileMenu (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
end

% --------------------------------------------------------------------
function PlotMenuItem_Callback(hObject, eventdata, handles)
    % hObject    handle to PlotMenuItem (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
end

% --------------------------------------------------------------------
function CloseMenuItem_Callback(hObject, eventdata, handles)
    % hObject    handle to CloseMenuItem (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    selection = questdlg(['Close ' get(handles.figure1, 'Name') '?'], ...
        ['Close ' get(handles.figure1, 'Name') '...'], ...
        'Yes', 'No', 'Yes');

    if strcmp(selection, 'No')
        return;
    end

    delete(handles.figure1);
end

% --- Executes on slider movement. slider1 is volume slider
% #ChangeSliceOrVolume#
function slider1_Callback(hObject, eventdata, handles)
    % hObject    handle to slider1 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)

    % Hints: get(hObject,'Value') returns position of slider
    %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

    % set slider activity
    handles.slider1_is_active = 1;
    handles.slider2_is_active = 0;
    % we are pushing slider value to figure
    handles.if_retrieve = 1;
    %refresh the frame
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.current_volume = round(handles.slider1.Value);
    handles.current_slice = round(handles.slider2.Value);
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
    % reset to default
    handles.if_retrieve = 0;
    guidata(hObject, handles);
end

% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to slider1 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject, 'BackgroundColor'), get(0, 'defaultUicontrolBackgroundColor'))
        set(hObject, 'BackgroundColor', [.9 .9 .9]);
    end

end

% --- Executes on slider movement.
% #ChangeSliceOrVolume#
function slider2_Callback(hObject, eventdata, handles)
    
    % hObject    handle to slider2 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)

    % Hints: get(hObject,'Value') returns position of slider
    %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

    % set slider activity
    handles.slider1_is_active = 0;
    handles.slider2_is_active = 1;
    % we are pushing slider value to figure
    handles.if_retrieve = 1;
    %refresh the frame
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.current_volume = round(handles.slider1.Value);
    handles.current_slice = round(handles.slider2.Value);
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
    % reset to default
    handles.if_retrieve = 0;
    guidata(hObject, handles);
end

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to slider2 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called

    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject, 'BackgroundColor'), get(0, 'defaultUicontrolBackgroundColor'))
        set(hObject, 'BackgroundColor', [.9 .9 .9]);
    end

end

% ----click on the axes to identify neuronal positions and update neuronal
% position in rest of the frames ------------------

function ButtonDown_Callback(hObject, eventdata, handles)

    [x, y] = getcurpt(handles.axes1);
    z = handles.current_slice;

    if strcmp(get(handles.figure1, 'selectionType'), 'alt')

        centroids = handles.neuronal_position{handles.current_volume, 1};
        neuron_idx = handles.neuronal_idx{handles.current_volume, 1};
        N = size(centroids, 2);
        missing_idx = find_missing_idx(neuron_idx);

        prompt = {'Enter neuron index:'};
        dlg_title = 'Neuronal identification';
        num_lines = 1;
        def = {num2str(missing_idx)};
        answer = inputdlg(prompt, dlg_title, num_lines, def);

        if ~isempty(answer)
            idx = str2num(answer{1});
            centroids(:, N + 1) = [x; y; z];
            neuron_idx(N + 1) = idx;

            handles.neuronal_position{handles.current_volume, 1} = centroids;
            handles.neuronal_idx{handles.current_volume, 1} = neuron_idx;
            handles.colorset = varycolor(max(neuron_idx));

            axes(handles.axes1);
            hold on;

            handles.points{N + 1} = text(x, y, num2str(neuron_idx(N + 1)));
            set(handles.points{N + 1}, 'Color', handles.colorset(neuron_idx(N + 1), :));
            set(handles.points{N + 1}, 'HorizontalAlignment', 'center');
            set(handles.points{N + 1}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownPoint_Callback'',gcbo,[],guidata(gcbo))');
            set(handles.points{N + 1}, 'fontsize', handles.fontsize);
            set(handles.points{N + 1}, 'FontWeight', 'bold');
            hObject = handles.points{N + 1};
            guidata(hObject, handles);

        end

    end

    if strcmp(get(handles.figure1, 'selectionType'), 'normal')

        centroids = handles.neuronal_position{handles.current_volume, 1};
        neuron_idx = handles.neuronal_idx{handles.current_volume, 1};
        N = size(centroids, 2);
        missing_idx = find_missing_idx(neuron_idx);

        def = {num2str(missing_idx)};

        if def{1} ~= 0
            answer = def;
        else
            answer{1} = num2str(1);
        end

        if ~isempty(answer)
            idx = str2num(answer{1});
            centroids(:, N + 1) = [x; y; z];
            neuron_idx(N + 1) = idx;

            handles.neuronal_position{handles.current_volume, 1} = centroids;
            handles.neuronal_idx{handles.current_volume, 1} = neuron_idx;
            handles.colorset = varycolor(max(neuron_idx));

            axes(handles.axes1);
            hold on;

            handles.points{N + 1} = text(x, y, num2str(neuron_idx(N + 1)));
            set(handles.points{N + 1}, 'Color', handles.colorset(neuron_idx(N + 1), :));
            set(handles.points{N + 1}, 'HorizontalAlignment', 'center');
            set(handles.points{N + 1}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownPoint_Callback'',gcbo,[],guidata(gcbo))');
            set(handles.points{N + 1}, 'fontsize', handles.fontsize);
            set(handles.points{N + 1}, 'FontWeight', 'bold');
            hObject = handles.points{N + 1};
            guidata(hObject, handles);

        end

    end

end

% ----click on the axes with "shift" to unlabel a neuron
% ------------------

function ButtonDownPoint_Callback(hObject, eventdata, handles)

    [x, y] = getcurpt(handles.axes1);
    z = handles.current_slice;
    w = 12; %weight factor for z resolution

    if strcmp(get(handles.figure1, 'selectionType'), 'extend')

        centroids = handles.neuronal_position{handles.current_volume, 1};
        neuron_idx = handles.neuronal_idx{handles.current_volume, 1};
        N = size(centroids, 2);

        distance_square = sum((centroids(1:2, :)' - repmat([x y], N, 1)).^2, 2); %the precision when click
        distance = sqrt(distance_square + w^2 * (centroids(3, :)' - z).^2);
        [~, k] = min(distance);
        neuron_idx(k) = [];
        centroids(:, k) = [];

        handles.neuronal_position{handles.current_volume, 1} = centroids;
        handles.neuronal_idx{handles.current_volume, 1} = neuron_idx;

        delete(hObject);

        handles.colorset = varycolor(max(neuron_idx));

        guidata(gcf, handles);

    end

end

% --------------------------------------------------------------------
function ExportMenuItem_Callback(hObject, eventdata, handles)
    % hObject    handle to ExportMenuItem (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    assignin('base', 'neuron_position_data', handles.neuronal_position);
    assignin('base', 'neuron_index_data', handles.neuronal_idx);
    assignin('base', 'ROI_data', handles.ROIposition);
    assignin('base', 'calcium_signal', handles.calcium_signals);
    assignin('base', 'neuron_boxes', handles.Boxes);
    SaveMenuItem_Callback(hObject, eventdata, handles);
end
% --------------------------------------------------------------------
function ImageMenuItem_Callback(hObject, eventdata, handles)
    % hObject    handle to ImageMenuItem (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
end
% --------------------------------------------------------------------
function LUTMenuItem_Callback(hObject, eventdata, handles)
    % hObject    handle to LUTMenuItem (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    answer = inputdlg({'low', 'high'}, 'Cancel to clear previous', 1, ...
        {num2str(handles.low), num2str(handles.high)});
    handles.low = str2double(answer{1});
    handles.high = str2double(answer{2});
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.hBoxes=Refresh_One_Frame(hObject,handles);
    guidata(hObject, handles);
end

% --- Executes on scroll wheel click while the figure is in focus.
% #ChangeSliceOrVolume#
function figure1_WindowScrollWheelFcn(hObject, eventdata, handles)
    % hObject    handle to figure1 (see GCBO)
    % eventdata  structure with the following fields (see FIGURE)
    %	VerticalScrollCount: signed integer indicating direction and number of clicks
    %	VerticalScrollAmount: number of lines scrolled for each click
    % handles    structure with handles and user data (see GUIDATA)

    %set wheel activity
    if eventdata.VerticalScrollCount > 0
        if handles.slider1_is_active % slider1 is the volume slider
            if handles.slider1.Value < handles.slider1.Max
                handles.previous_volume = handles.current_volume;
                handles.current_volume = handles.current_volume + 1;
                handles.previous_slice=handles.current_slice;
            else
                handles.slider1.Value = handles.slider1.Max;
            end
        elseif handles.slider2_is_active
            if handles.slider2.Value < handles.slider2.Max
                handles.previous_slice = handles.current_slice;
                handles.current_slice = handles.current_slice + 1;
                handles.previous_volume=handles.current_volume;
            else
                handles.slider2.Value = handles.slider2.Max;
            end
        end
        
    elseif eventdata.VerticalScrollCount < 0
        if handles.slider1_is_active % slider1 is the volume slider
            if handles.slider1.Value > handles.slider1.Min
                handles.previous_volume = handles.current_volume;
                handles.current_volume = handles.current_volume - 1;
                handles.previous_slice=handles.current_slice;
            else 
                handles.slider1.Value = handles.slider1.Min;
            end
        elseif handles.slider2_is_active
            if handles.slider2.Value > handles.slider2.Min
                handles.previous_slice = handles.current_slice;
                handles.current_slice = handles.current_slice - 1;
                handles.previous_volume=handles.current_volume;
            else 
                handles.slider2.Value = handles.slider2.Min;
            end
        end

    end

    % set state of retrieve or push in figure
    handles.if_retrieve = 0;
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
    %guidata(hObject,handles);
end

% --------------------------------------------------------------------
function ButtonDownROI_Callback(hObject, eventdata, handles)

    initialrect = get(handles.ROI{handles.current_volume, handles.current_slice}, 'Position');
    initialrect(2) = handles.image_height - initialrect(2);
    finalrect = dragrect(initialrect);
    finalrect(2) = handles.image_height - finalrect(2);
    set(handles.ROI{handles.current_volume, handles.current_slice}, 'Position', finalrect);
    handles.ROIposition{handles.current_volume, handles.current_slice} = finalrect;
    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function zoom_Callback(hObject, eventdata, handles)
    % hObject    handle to zoom (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)

    checked = get(hObject, 'Checked');

    if strcmp(checked, 'off')
        set(hObject, 'Checked', 'on');
        zoom on;

    else
        set(hObject, 'Checked', 'off');
        zoom off;

    end

    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function Enhance_contrast_Callback(hObject, eventdata, handles)
    % hObject    handle to Enhance_contrast (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    checked = get(hObject, 'Checked');

    if strcmp(checked, 'off')
        set(hObject, 'Checked', 'on');
        handles.contrast_enhancement = 1;
    else
        set(hObject, 'Checked', 'off');
        handles.contrast_enhancement = 0;
    end

    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function Display_reference_Callback(hObject, eventdata, handles)
    % hObject    handle to Display_reference (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)

    answer = inputdlg({'reference volume to be displayed'}, 'Cancel to clear previous', 1, ...
        {num2str(1)});

    handles.reference_volume = str2double(answer{1});

    figure ('Numbertitle', 'off', 'Name', 'Reference Slide Display', ...
        'Position', [100 100 handles.axel_width handles.axel_height + 30], ...
        'WindowScrollWheelFcn', {@Reference_Display_ScrollFcn, hObject, handles});

    %set(gcf,'WindowScrollWheelFcn', 'whole_brain_imaging(''Reference_Display_ScrollFcn'',gcbo,[],guidata(gcbo))');

    set(gca, ...
        'Visible', 'off', ...
        'Units', 'pixels', ...
        'Position', [0 0 handles.axel_width handles.axel_height]);

    cla;

    imagesc(handles.img_stack{handles.reference_volume, 1}(:, :, handles.current_slice), [handles.low handles.high]);
    colormap(gray);
    axis equal;
    axis off;

    text1 = uicontrol(gcf, 'Style', 'text');
    set(text1, 'Units', 'pixels');
    set(text1, 'Position', [0 0 + handles.axel_height - 10 handles.axel_width 30]);
    set(text1, 'HorizontalAlignment', 'center');
    set(text1, 'String', strcat(num2str(handles.current_slice), '/', num2str(handles.slices), ' z slices'));

    N = size((handles.neuronal_position{handles.reference_volume, 1}), 2);
    neuron_idx = handles.neuronal_idx{handles.reference_volume, 1};
    handles.colorset = varycolor(max(neuron_idx));

    for j = 1:N

        x = handles.neuronal_position{handles.reference_volume, 1}(1, j);
        y = handles.neuronal_position{handles.reference_volume, 1}(2, j);
        z = round(handles.neuronal_position{handles.reference_volume, 1}(3, j));

        if round(z) == handles.current_slice

            hold on;
            h = text(x, y, num2str(neuron_idx(j)));
            set(h, 'HorizontalAlignment', 'center');
            set(h, 'Color', handles.colorset(neuron_idx(j), :));
            set(h, 'FontWeight', 'bold');
        end

    end

    guidata(hObject, handles);
end
function Reference_Display_ScrollFcn(src, eventdata, hObject, handles)

    h = get(gcf, 'Children');

    text1 = h(1);

    content = get(text1, 'String');

    idx = strfind(content, '/');

    current_slice = str2double(content(1:idx - 1));

    if eventdata.VerticalScrollCount > 0

        if current_slice < handles.slices
            current_slice = current_slice + 1;
        end

    elseif eventdata.VerticalScrollCount < 0

        if current_slice > 1
            current_slice = current_slice - 1;
        end

    end

    cla;
    imagesc(handles.img_stack{handles.reference_volume, 1}(:, :, current_slice), [handles.low handles.high]);
    colormap(gray);
    axis equal;
    axis off;
    set(handles.axes1, 'Visible', 'off');

    set(text1, 'String', strcat(num2str(current_slice), '/', num2str(handles.slices), ' z slices'));

    N = size((handles.neuronal_position{handles.reference_volume, 1}), 2);
    neuron_idx = handles.neuronal_idx{handles.reference_volume, 1};
    handles.colorset = varycolor(max(neuron_idx));

    for j = 1:N

        x = handles.neuronal_position{handles.reference_volume, 1}(1, j);
        y = handles.neuronal_position{handles.reference_volume, 1}(2, j);
        z = round(handles.neuronal_position{handles.reference_volume, 1}(3, j));

        if round(z) == current_slice

            hold on;
            h = text(x, y, num2str(neuron_idx(j)));
            set(h, 'HorizontalAlignment', 'center');
            set(h, 'Color', handles.colorset(neuron_idx(j), :));
            set(h, 'FontWeight', 'bold');
        end

    end

end

% --------------------------------------------------------------------
function fontsize_Callback(hObject, eventdata, handles)
    % hObject    handle to fontsize (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    dlg_title = 'font size';
    prompt = {'font size'};
    num_lines = 1;
    marks = inputdlg(prompt, dlg_title, num_lines, {'10'});
    handles.fontsize = str2num(marks{1}(1, :));
    set(handles.slider1, 'Value', handles.current_volume);
    set(handles.slider2, 'Value', handles.current_slice);

    set(handles.text1, 'String', strcat(num2str(handles.current_volume), '/', num2str(handles.volumes), ' volumes; ', num2str(handles.current_slice), '/', num2str(handles.slices), ' z slices'));
    axes(handles.axes1);
    cla;

    if handles.contrast_enhancement
        img = imagesc(contrast_enhancement(handles.img_stack{handles.current_volume, 1}(:, :, handles.current_slice)), [handles.low handles.high]);
    else
        img = imagesc(handles.img_stack{handles.current_volume, 1}(:, :, handles.current_slice), [handles.low handles.high]);
    end

    colormap(gray);
    set(handles.axes1, 'Visible', 'off');
    set(img, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDown_Callback'',gcbo,[],guidata(gcbo))');

    if ~isempty(handles.neuronal_position{handles.current_volume, 1})

        N = size((handles.neuronal_position{handles.current_volume, 1}), 2);
        neuron_idx = handles.neuronal_idx{handles.current_volume, 1};
        handles.colorset = varycolor(max(neuron_idx)); %a bug fixed by fanchan

        for j = 1:N

            x = handles.neuronal_position{handles.current_volume, 1}(1, j);
            y = handles.neuronal_position{handles.current_volume, 1}(2, j);
            z = round(handles.neuronal_position{handles.current_volume, 1}(3, j));

            if round(z) == handles.current_slice

                hold on;
                handles.points{j} = text(x, y, num2str(neuron_idx(j)));
                set(handles.points{j}, 'Color', handles.colorset(neuron_idx(j), :));
                set(handles.points{j}, 'HorizontalAlignment', 'center');
                set(handles.points{j}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownPoint_Callback'',gcbo,[],guidata(gcbo))');
                set(handles.points{j}, 'fontsize', handles.fontsize);
                set(handles.points{j}, 'FontWeight', 'bold');
            end

        end

    end

    handles.hp = impixelinfo;

    set(handles.hp, 'Position', [0 44 + handles.axel_height + 2, 300, 20]);
    guidata(hObject, handles);
end

% --- Executes on key press with focus on figure1 or any of its controls.
%#ChangeSliceOrVolume#
function figure1_WindowKeyPressFcn(hObject, eventdata, handles)
    % hObject    handle to figure1 (see GCBO)
    % eventdata  structure with the following fields (see MATLAB.UI.FIGURE)
    %	Key: name of the key that was pressed, in lower case
    %	Character: character interpretation of the key(s) that was pressed
    %	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
    % handles    structure with handles and user data (see GUIDATA)
   
    % set state of retrieve or push in figure
    handles.if_retrieve = 0;

    switch eventdata.Key
        case 'a'
            handles.previous_slice = handles.current_slice;
            handles.current_slice = handles.current_slice - 1;
            if handles.current_slice < 1
                handles.current_slice=1;
            end
            handles.previous_volume=handles.current_volume;
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
        case 's'
            handles.previous_slice = handles.current_slice;
            handles.current_slice = handles.current_slice + 1;
            if handles.current_slice > handles.slices
                handles.current_slice = handles.slices;
            end
            handles.previous_volume=handles.current_volume;
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
        case 'z'
            handles.previous_volume = handles.current_volume;
            handles.current_volume = handles.current_volume - 1;
            if handles.current_volume < 1
                handles.current_volume=1;
            end
            handles.previous_slice=handles.current_slice;
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
        case 'x'
            handles.previous_volume = handles.current_volume;
            handles.current_volume = handles.current_volume + 1;
            if handles.current_volume > handles.volumes;
                handles.current_volume = handles.volumes;
            end
            handles.previous_slice=handles.current_slice;
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
            % with 5 steps
        case 'd'
            handles.previous_slice = handles.current_slice;
            handles.current_slice = handles.current_slice - 5;
            if handles.current_slice < 1
                handles.current_slice=1;
            end
            handles.previous_volume=handles.current_volume;
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
        case 'f'
            handles.previous_slice = handles.current_slice;
            handles.current_slice = handles.current_slice + 5;
            if handles.current_slice > handles.slices
                handles.current_slice = handles.slices;
            end
            handles.previous_volume=handles.current_volume;
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
        case 'c'
            handles.previous_volume = handles.current_volume;
            handles.current_volume = handles.current_volume - 5;
            if handles.current_volume < 1
                handles.current_volume=1;
            end
            handles.previous_slice=handles.current_slice;
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
        case 'v'
            handles.previous_volume = handles.current_volume;
            handles.current_volume = handles.current_volume + 5;
            if handles.current_volume > handles.volumes;
                handles.current_volume = handles.volumes;
            end
            handles.previous_slice=handles.current_slice;
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
            
        case {'equal', 'hyphen'}
            [handles.axes1.XLim, handles.axes1.YLim] = zoom_shortcut(eventdata.Key, handles.axes1.XLim, handles.axes1.YLim, [0.5, 1024.5]);
            handles.hBoxes=Refresh_One_Frame(hObject, handles);
            guidata(hObject, handles);
        case {'leftarrow', 'rightarrow', 'uparrow', 'downarrow', 'j', 'k', 'l', 'i'}
            [handles.axes1.XLim, handles.axes1.YLim] = pan_shortcut(eventdata.Key, handles.axes1.XLim, handles.axes1.YLim, [0.5, 1024.5]);
            guidata(hObject, handles);
    end
    
end

% ---------------------------------------------------------------
function slider2_KeyPressFcn(hObject, eventdata, handles)
    % hObject    handle to figure1 (see GCBO)
    % eventdata  structure with the following fields (see MATLAB.UI.FIGURE)
    %	Key: name of the key that was pressed, in lower case
    %	Character: character interpretation of the key(s) that was pressed
    %	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
    % handles    structure with handles and user data (see GUIDATA)
    if eventdata.Key == 'q'
        handles.slider1_is_active = 0;
        handles.slider2_is_active = 1;
    end
    
    if eventdata.Key == 'w'
        handles.slider1_is_active = 1;
        handles.slider2_is_active = 0;
    end
    if strcmp(eventdata.Key,'leftarrow')
    end
    if strcmp(eventdata.Key, 'rightarrow')
    end
    

    guidata(hObject, handles);
end

% --- Executes on key press with focus on slider1 and none of its controls.
function slider1_KeyPressFcn(hObject, eventdata, handles)
    % hObject    handle to slider1 (see GCBO)
    % eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
    %	Key: name of the key that was pressed, in lower case
    %	Character: character interpretation of the key(s) that was pressed
    %	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
    % handles    structure with handles and user data (see GUIDATA)
    if eventdata.Key == 'q'
        handles.slider1_is_active = 0;
        handles.slider2_is_active = 1;
    end

    if eventdata.Key == 'w'
        handles.slider1_is_active = 1;
        handles.slider2_is_active = 0;
    end
    if strcmp(eventdata.Key,'leftarrow')
    end
    if strcmp(eventdata.Key,'rightarrow')
    end

    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function tracking_all_Callback(hObject, eventdata, handles)
    % hObject    handle to tracking_all (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    answer = inputdlg({'Start frame', 'End frame'}, '', 1);
    istart = str2double(answer{1});
    iend = str2double(answer{2});

    if istart == handles.volumes
        printf('Don''t use the last frame as the start frame.');
        return;
    end

    for i = (istart + 1):iend
        imgStack_pre = handles.img_stack{i - 1, 1};
        imgStack = handles.img_stack{i, 1};

        if ~isempty(handles.neuronal_position{i - 1, 1})

            if handles.contrast_enhancement

                registed_centers = identify_neuronal_position(contrast_enhancement(imgStack), handles.ROIposition(i, 1:handles.slices), ...
                    imgStack_pre, handles.ROIposition(i - 1, 1:handles.slices), handles.neuronal_position{i - 1, 1});
            else

                registed_centers = identify_neuronal_position(imgStack, handles.ROIposition(i, 1:handles.slices), ...
                    imgStack_pre, handles.ROIposition(i - 1, 1:handles.slices), handles.neuronal_position{i - 1, 1});
            end

            if ~isempty(registed_centers)
                handles.neuronal_position{i, 1} = adjust_abnormal_centroids(registed_centers, handles.image_width, handles.image_height);
                handles.neuronal_idx{i, 1} = handles.neuronal_idx{i - 1, 1};
            end

        else

            disp('Please proof read the start volume.');

        end

        fprintf('Finish the %dth volume.\n', i);
    end

    msgbox('Tracking Completed!');
    guidata(hObject, handles);
end

% --------------------------------------------------------------------
    function hBoxes=Refresh_One_Frame(hObject, handles)
    % hObject    handle to figure1 (see GCBO)
    % handles    structure with handles and user data (see GUIDATA)
    % Refresh current frame display.
        
    % if we are retrieving or pushing
    if handles.if_retrieve% retrive slider value from figure
        handles.current_volume = round(handles.slider1.Value);
        handles.current_slice = round(handles.slider2.Value);
    else % set value change to figure
        set(handles.slider1, 'Value', handles.current_volume);
        set(handles.slider2, 'Value', handles.current_slice);
    end
    % save hBoxes from "memory" to "harddisk"
    pv=handles.previous_volume;
    ps=handles.previous_slice;
    if ~isempty(handles.hBoxes)
        Boxes=make_a_BOX;
        for b = 1:length(handles.hBoxes)
            if isvalid(handles.hBoxes(b))
                Boxes(end+1)=handles.hBoxes(b).UserData;
                Boxes(end).Position=handles.hBoxes(b).Position;
                handles.hBoxes(b).delete;
            end
        end
        handles.Boxes{pv,ps} = Boxes;
    end
    % reset the handle
    handles.hBoxes = images.roi.Rectangle.empty;
    % store current xy lim
    xlim = handles.axes1.XLim;
    ylim = handles.axes1.YLim;

    set(handles.text1, 'String', strcat(num2str(handles.current_volume), '/', num2str(handles.volumes), ' volumes; ', num2str(handles.current_slice), '/', num2str(handles.slices), ' z slices'));
    axes(handles.axes1);
    cla;

    if handles.contrast_enhancement
        img = imagesc(contrast_enhancement(handles.img_stack{handles.current_volume, 1}(:, :, handles.current_slice)), [handles.low handles.high]);
    else
        img = imagesc(handles.img_stack{handles.current_volume, 1}(:, :, handles.current_slice), [handles.low handles.high]);
    end
    imgMatrix=handles.img_stack{handles.current_volume, 1}(:, :, handles.current_slice);
    if handles.CmapJet==0
        colormap(gray);
    else
        colormap(jet);
    end
    set(handles.axes1, 'Visible', 'off');
    %restore xy lim
    handles.axes1.XLim = xlim;
    handles.axes1.YLim = ylim;
    % set neuron-label buttondown callback or neuron-bound buttondown callback or assign-outlier-box button down
    switch handles.ButtonDownFunc
        case 'point'
            set(img, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDown_Callback'',gcbo,[],guidata(gcbo))');
        case 'bound_neuron'
            set(img, 'ButtonDownFcn', 'whole_brain_imaging(''Bound_neuron_ButtonDown_Callback'',gcbo,[],guidata(gcbo))');
            %turn off zoom and pan for convenience
            if strcmp(handles.zoom.Checked, 'on')
                zoom 'off';
                handles.zoom.Checked = 'off';
            end
            if strcmp(handles.Pan.Checked, 'on')
                pan 'off';
                handles.Pan.Checked = 'off';
            end
        case 'Assign_index_to_box'
            set(img, 'ButtonDownFcn', 'whole_brain_imaging(''Assign_index_to_box_idx_ButtonDown_Callback'',gcbo,[],guidata(gcbo))');
        otherwise
    end

    %% display neuronal index
    %red channel

    if ~isempty(handles.neuronal_position{handles.current_volume, 1})

        N = size((handles.neuronal_position{handles.current_volume, 1}), 2);
        neuron_idx = handles.neuronal_idx{handles.current_volume, 1};
        handles.colorset = varycolor(max(neuron_idx)); %a bug fixed by fanchan

        for j = 1:N

            x = handles.neuronal_position{handles.current_volume, 1}(1, j);
            y = handles.neuronal_position{handles.current_volume, 1}(2, j);
            z = round(handles.neuronal_position{handles.current_volume, 1}(3, j));

            if round(z) == handles.current_slice

                hold on;

                handles.points{j} = text(x, y, num2str(neuron_idx(j)));
                set(handles.points{j}, 'Color', handles.colorset(neuron_idx(j), :));
                set(handles.points{j}, 'HorizontalAlignment', 'center');
                set(handles.points{j}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownPoint_Callback'',gcbo,[],guidata(gcbo))');
                set(handles.points{j}, 'fontsize', handles.fontsize); %newly added.
                set(handles.points{j}, 'FontWeight', 'bold');
                % Turn off point display when we are displaying box
                if handles.DispBox
                    handles.points{j}.Visible = 'off';
                end

            end

        end

    end

    %green channel

    if ~isempty(handles.neuronal_position{handles.current_volume, 2})

        for j = 1:N

            x = handles.neuronal_position{handles.current_volume, 2}(1, j);
            y = handles.neuronal_position{handles.current_volume, 2}(2, j);
            z = round(handles.neuronal_position{handles.current_volume, 2}(3, j));

            if round(z) == handles.current_slice

                hold on;

                handles.points{j} = text(x, y, num2str(neuron_idx(j)));
                set(handles.points{j}, 'Color', handles.colorset(neuron_idx(j), :));
                set(handles.points{j}, 'HorizontalAlignment', 'center');
                set(handles.points{j}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownPoint_Callback'',gcbo,[],guidata(gcbo))');
                set(handles.points{j}, 'fontsize', handles.fontsize); %newly added.
                set(handles.points{j}, 'FontWeight', 'bold');
            end

        end

    end

    %%ROI

    if ~isempty(handles.ROIposition{handles.current_volume, handles.current_slice})
        rect = handles.ROIposition{handles.current_volume, handles.current_slice};
        handles.ROI{handles.current_volume, handles.current_slice} = rectangle('Curvature', [0 0], 'Position', rect, 'EdgeColor', 'y');
        set(handles.ROI{handles.current_volume, handles.current_slice}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownROI_Callback'',gcbo,[],guidata(gcbo))');

    elseif ~isempty(handles.ROIposition{max(handles.current_volume - 1, 1), handles.current_slice})
        rect = handles.ROIposition{max(handles.current_volume - 1, 1), handles.current_slice};
        handles.ROI{handles.current_volume, handles.current_slice} = rectangle('Curvature', [0 0], 'Position', rect, 'EdgeColor', 'y');
        handles.ROIposition{handles.current_volume, handles.current_slice} = rect;
        set(handles.ROI{handles.current_volume, handles.current_slice}, 'ButtonDownFcn', 'whole_brain_imaging(''ButtonDownROI_Callback'',gcbo,[],guidata(gcbo))');
    end
    
    %online pixel intensity calculation
    handles.hp = impixelinfo;
    handles.refreshed_times = handles.refreshed_times + 1;
    set(handles.hp, 'Position', [0 44 + handles.axel_height + 2, 300, 20]);
    
    %% display neuron box on this slice.
    
    if handles.DispBox
        % first we save handle of neuron_box to the "harddisk", and clear handle of neuron_box
        % then we retrieve handle of neuron_box from "harddisk" to "memory"
        hBoxes = images.roi.Rectangle.empty;
        cv=handles.current_volume;
        cs=handles.current_slice;
        % determine the color of each pixel mapped to colormap
        cmap=colormap;
        cl=caxis;
        % determine the fontsize
        fontsize=12;
        fontsize_zoom_factor=6;
        zoom_fold=size(img.CData,2)/(handles.axes1.XLim(2)-handles.axes1.XLim(1));
        fontsize=fix(fontsize+zoom_fold/fontsize_zoom_factor);
        fontsize_per_unit=12/16; % this is calculated with XLim [0 1024], thus when fontsize is 12, it occupies 16 pixels.

        if ~isempty(handles.Boxes{cv,cs})
            for b=1:length(handles.Boxes{cv,cs})
                hBoxes(end + 1) = images.roi.Rectangle(gca, 'Position', handles.Boxes{cv,cs}(b).Position, 'Color', 'm', 'LineWidth', 1, 'FaceAlpha', 0);
                hBoxes(end).UserData=handles.Boxes{cv,cs}(b);
            end
        end
        handles.hBoxes=hBoxes;
        % if we have box to display
        if ~isempty(handles.hBoxes)
            points = zeros(3, 302); % max number of neurons is 302
            if ~isempty(handles.neuronal_position{handles.current_volume, 1})
                points_seq = find(handles.neuronal_position{handles.current_volume, 1}(3, :) == handles.current_slice);
                points(1:2, 1:length(points_seq)) = handles.neuronal_position{handles.current_volume, 1}(1:2, points_seq);
                points(3, 1:length(points_seq)) = handles.neuronal_idx{handles.current_volume}(points_seq);
            end
            points = points';
            
            % check for duplicated or empty indexes  
            consipicuousFlag=ones(length(handles.hBoxes),1); % duplicated or empty index would be 1.
            for b = 1:length(handles.hBoxes)
                if ~isempty(handles.hBoxes(b).UserData.idx)
                    box_idx(b)=handles.hBoxes(b).UserData.idx;
                end
            end
            if ~isempty(handles.hBoxes(b).UserData.idx)
                [C,IA,IC]=unique(box_idx,'stable');% find unique indexes of this slice
                BinEdges=0.5:1:max(IC)+0.5;
                [counts,edges]=histcounts(IC,BinEdges); % get the index of repetitions
                IA(ceil(edges(counts>1)))=[]; 
                consipicuousFlag(IA)=0;
                consipicuousFlag(box_idx==0)=1; % empty index
                if ~isempty(handles.neuron2find)
                    for i=1:length(handles.neuron2find)
                        consipicuousFlag(box_idx==handles.neuron2find(i))=1;
                    end
                end
            end

            % we only display boxes that on this slice: boxes
            for b = 1:length(handles.hBoxes)
                % judge if there is a labeled neuron (from neuron_position_data) locating inside the box.
                tf = inROI(handles.hBoxes(b), points(:, 1), points(:, 2));
                handles.hBoxes(b).UserData.isinlier = sum(tf);
                if ~isempty(points(tf, 3))
                    handles.hBoxes(b).UserData.idx = points(tf, 3);
                end
                x = handles.hBoxes(b).Position(1) + handles.hBoxes(b).Position(3) / 2;
                y = handles.hBoxes(b).Position(2) + handles.hBoxes(b).Position(4) / 2;
                if handles.DispBoxID==1 % Display neuron id inside Box
                    t=text(x, y, num2str(handles.hBoxes(b).UserData.idx), 'HorizontalAlignment', 'center', ...
                    'Margin',0.1,'FontSize', fontsize, 'FontWeight', 'bold');
                    text_region=t.Extent; %  [left bottom width height]
                    text_region(1:2)=floor(text_region(1:2));
                    text_region(3:4)=ceil(text_region(3:4));
                    % calculate the complementray color of pixels inside the box
                    text_region_mean=mean(mean(img.CData(text_region(2) - text_region(4):text_region(2),...
                        text_region(1):text_region(1) + text_region(3))));
                    text_region_mean(text_region_mean>cl(2)-cl(1))=cl(2);
                    text_region_mean(text_region_mean<=cl(1))=cl(1)+0.001;
                    color_rank_inside=ceil((double(text_region_mean)-cl(1))/(cl(2)-cl(1))*size(cmap,1));
                    % set text color and its background
                    text_col=1-cmap(color_rank_inside,:);
                    t.Color = text_col;
                    if fix(zoom_fold/4)
                        t.BackgroundColor=cmap(color_rank_inside,:);
                    end                  
                    set(t, 'ButtonDownFcn', 'whole_brain_imaging(''Assign_index_to_box_idx_ButtonDown_Callback'',gcbo,[],guidata(gcbo))');
                elseif handles.ShowMeanVal==1 % Display mean value of pixels inside Box
                    PixelsInBox=imgMatrix(round(handles.hBoxes(b).Position(2) : handles.hBoxes(b).Position(2)+handles.hBoxes(b).Position(4)),...
                    round(handles.hBoxes(b).Position(1) : handles.hBoxes(b).Position(1)+handles.hBoxes(b).Position(3)));
                    MeanVal=round(mean(mean(PixelsInBox)));
                    text(x, y, num2str(MeanVal), 'HorizontalAlignment', 'center', ...
                        'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
                end
                if consipicuousFlag(b) % boxes with duplicated or empty index
                    handles.hBoxes(b).Color = 'y';
                    handles.hBoxes(b).LineWidth=4;
                end
                switch handles.ButtonDownFunc
                    case 'bound_neuron'
                    otherwise
                        handles.hBoxes(b).InteractionsAllowed = 'none';
                        handles.hBoxes(b).FaceSelectable = 0;
                        handles.hBoxes(b).Selected = 0;
                end
            end
        end
    end
    hBoxes=handles.hBoxes;
    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function Pan_Callback(hObject, eventdata, handles)
    % hObject    handle to Pan (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    checked = get(hObject, 'Checked');

    if strcmp(checked, 'off')
        set(hObject, 'Checked', 'on');
        pan on;

    else
        set(hObject, 'Checked', 'off');
        pan off;

    end

    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function Bound_neuron_Callback(hObject, eventdata, handles)
    % hObject    handle to Bound_neuron (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)

    checked = get(hObject, 'Checked');

    if strcmp(checked, 'off')
        set(hObject, 'Checked', 'on');

        % we also need to turn off Assign_index_to_box
        %   first we need to find the handle
        if strcmp(handles.ButtonDownFunc, 'Assign_index_to_box')
            menu_handle = handles.Annotation.Children;
            for i = 1:length(menu_handle)
                if strcmp(menu_handle(i).Tag, 'Assign_index_to_box')
                    % we have found assign outlier box handle
                    menu_handle = menu_handle(i);
                    break;
                end
            end
            menu_handle.Checked = 'off';
        end

        % we also need to turn on Display neruon box
        if handles.DispBox == 0
            handles.Display_neuron_box.Checked = 'on';
            handles.DispBox = 1;
        end

        % set ButtonDownFunc
        handles.ButtonDownFunc = 'bound_neuron';
        % set state of retrieve or push in figure
        handles.if_retrieve = 0;
        handles.bounding_neuron=1;
        handles.previous_slice = handles.current_slice;
        handles.previous_volume = handles.current_volume;
        handles.hBoxes=Refresh_One_Frame(hObject, handles);
    else
        set(hObject, 'Checked', 'off');
        handles.ButtonDownFunc = 'point';
        % set state of retrieve or push in figure
        handles.if_retrieve = 0;
        cv = handles.current_volume;
        cs = handles.current_slice;
        handles.bounding_neuron=0;
        handles.previous_slice = handles.current_slice;
        handles.previous_volume = handles.current_volume;
        handles.hBoxes=Refresh_One_Frame(hObject, handles);
    end

    guidata(hObject, handles);
end

function Bound_neuron_ButtonDown_Callback(hObject, eventdata, handles)
    %% start to bound neurons
    % get currentpoint to imediately draw
    % assign UserData to hBox, which is a struct has fields of:
    % UserData.isinlier,
    % UserData.idx --> if a box is inlier, then we assign its pairing neuron index;
    %                  if a box is outlier, then set to empty.
    % UserData.identifier --> for box-editting overwrite
    % UserData.slice
    cp = get(gca, 'currentpoint');
    cp = [cp(1, 1) cp(1, 2)];
    hBox = images.roi.Rectangle(gca, 'Color', 'y', 'LineWidth', 1, 'FaceAlpha', 0);
    beginDrawingFromPoint(hBox, cp);
    % make this bounding valid, line of box should >= 2.
    if hBox.Position(3) * hBox.Position(4) > 3
        handles.box_nxt_id = handles.box_nxt_id + 1;
        box_centroid = hBox.Position(:, 1:2) + hBox.Position(:, 3:4) / 2; % the centroid of this box, box_centroid(1:2) correspond to x, y
        seq_of_box_pair_neu = [];

        if ~isempty(handles.neuronal_position{handles.current_volume, 1})
            neu_clicked_seq_slice = find(handles.neuronal_position{handles.current_volume, 1}(3, :) == handles.current_slice);
            num_of_neurons_on_slice = numel(neu_clicked_seq_slice);
            % only get neurons on this slice
            %   row 1 & row 2 are x-y coordinates, row 3 is the original sequence on hanles.neuronal_position
            %   'neuron_on_slice' has:
            %       row 1: x1 x2 x3...lenght
            %       row 2: y1 y2 y3...
            %       row 3: seq1 seq2 seq3...
            neuron_on_slice = handles.neuronal_position{handles.current_volume, 1}(:, neu_clicked_seq_slice);
            neuron_on_slice(3, :) = neu_clicked_seq_slice;
            % calculate distence
            if neuron_on_slice
                tf = inROI(hBox, neuron_on_slice(1, :), neuron_on_slice(2, :)); % a vetor that if neurons lie inside box
                in_box_neuron_cen = transpose(neuron_on_slice(1:2, tf));
                distance2 = sum((repmat(box_centroid, sum(tf), 1) - in_box_neuron_cen).^2, 2);
                [~, local_seq] = min(distance2); % sequence of min distance of neuron(s) inside box
                seq_of_box_pair_neu = neuron_on_slice(3, tf);
                seq_of_box_pair_neu = seq_of_box_pair_neu(local_seq);
            end

        end

        if ~isempty(seq_of_box_pair_neu)
            hBox.UserData.isinlier = 1;
            hBox.UserData.idx = handles.neuronal_idx{handles.current_volume}(seq_of_box_pair_neu);
            hBox.UserData.slice = handles.current_slice;
        else
            hBox.UserData.isinlier = 0;
            hBox.UserData.idx = [];
            hBox.UserData.slice = handles.current_slice;
        end

        num_boxes = length(handles.Boxes{handles.current_volume});
        hBox.UserData.identifier = handles.box_nxt_id;
        hBox.UserData.Position = hBox.Position;
        handles.hBoxes(end + 1) = hBox;
    end

    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function Export_all_data_Callback(hObject, eventdata, handles)
    % hObject    handle to Export_all_data (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    handles.img_stack = [];
    assignin('base', 'Whole_data', handles);
end

% --------------------------------------------------------------------
function Display_neuron_box_Callback(hObject, eventdata, handles)
    % hObject    handle to Display_neuron_box (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    ischecked = get(hObject, 'Checked');

    if strcmp(ischecked, 'off')
        set(hObject, 'Checked', 'on');
        handles.DispBox = 1;
    elseif strcmp(ischecked, 'on')
        set(hObject, 'Checked', 'off')
        handles.DispBox = 0;
        % we also need to turn off Assign_index_to_box
        switch handles.ButtonDownFunc
            case 'Assign_index_to_box'
                menu_handle = handles.Annotation.Children;

                for i = 1:length(menu_handle)

                    if strcmp(menu_handle(i).Tag, 'Assign_index_to_box')
                        %we have found assgin outlier box menu_handle.
                        menu_handle = menu_handle(i);
                        break;
                    end

                end

                menu_handle.Checked = 'off';
            case 'bound_neuron'
                handles.Bound_neuron.Checked = 'off';
            otherwise
        end

        % set state of retrieve or push in figure
        handles.ButtonDownFunc = 'point';
        handles.if_retrieve = 0;
        handles.previous_slice = handles.current_slice;
        handles.previous_volume = handles.current_volume;
        handles.hBoxes=Refresh_One_Frame(hObject, handles);
    end

    guidata(hObject, handles);
end


% get handles of neuron_box  (like memory) from stored variavles(like harddisk)
function grab_handle_of_neuron_box_from_Boxes(hObject,handles)
 
    setappdata(handles,'hBoxes',hBoxes);
    guidata(hObject, handles);
end
%% make a empty BOX struct
function BOX = make_a_BOX()
    BOX = struct('isinlier', {}, 'idx', {}, 'slice', {}, 'identifier', {}, 'Position', {});
end

% --------------------------------------------------------------------
% handles    structure with handles and user data (see GUIDATA)

function Assign_index_to_box_Callback(hObject, eventdata, handles)
    checked = get(hObject, 'Checked');

    if strcmp(checked, 'off')
        set(hObject, 'Checked', 'on');
        % we also need to turn on neuron box display
        if handles.DispBox == 0
            handles.Display_neuron_box.Checked = 'on';
            handles.DispBox = 1;
        end
        if handles.DispBoxID == 0
            handles.Display_BOX_ID.Checked = 'on';
            handles.DispBoxID = 1;
        end
        % we also need to turn off bound_neuron
        if strcmp(handles.ButtonDownFunc, 'bound_neuron')
            handles.Bound_neuron.Checked = 'off';
        end

        % set ButtonDownFunc
        handles.ButtonDownFunc = 'Assign_index_to_box';
        handles.previous_slice = handles.current_slice;
        handles.previous_volume = handles.current_volume;
        handles.hBoxes=Refresh_One_Frame(hObject, handles);
    else
        set(hObject, 'Checked', 'off');
        handles.ButtonDownFunc = 'point';
        handles.previous_slice = handles.current_slice;
        handles.previous_volume = handles.current_volume;
        handles.hBoxes=Refresh_One_Frame(hObject, handles);
    end

    guidata(hObject, handles);
end

function Assign_index_to_box_idx_ButtonDown_Callback(hObject, eventdata, handles)
    %% start to assign neuron index to outlier box
    % !!! Caution:this function must be used inside an outlier box !!!

    [x, y] = getcurpt(handles.axes1);
    inside_box_flag = 0;
    % right click, assign an arbitary number
    if strcmp(get(handles.figure1, 'selectionType'), 'normal')
        default(1) = convertCharsToStrings(num2str(handles.last_outlier_box_idx));
        default(2) = handles.last_outlier_box_choice;
        cp = handles.figure1.CurrentPoint;
        answer = editdialog(cp, default);

        if ~isempty(answer(1))
            NewIdx = str2num(answer(1));
            if ~isempty(handles.hBoxes)
                for b = 1:length(handles.hBoxes)
                    % we have clicked inside a box
                    if inROI(handles.hBoxes(b), x, y)
                        OldIdx=handles.hBoxes(b).UserData.idx;
                        handles.hBoxes(b).UserData.idx = NewIdx;
                        inside_box_flag = 1;
                        handles.last_outlier_box_idx = NewIdx;
                        handles.last_outlier_box_choice=answer(2);
                        break;
                    end
                end
                handles.Boxes = SearchNeuronBoxIndex(NewIdx,OldIdx,handles.Boxes,handles.current_volume,handles.current_slice,answer(2));
            else
                % we don't have box on this slice
                msgbox('There is no box on this slice!');
            end

        end

        if inside_box_flag
            axes(handles.axes1);
            hold on;
            x = handles.hBoxes(b).Position(1) + handles.hBoxes(b).Position(3) / 2;
            y = handles.hBoxes(b).Position(2) + handles.hBoxes(b).Position(4) / 2;
            idx_text = text(x, y, num2str(NewIdx));
            set(idx_text, 'Color', 'k');
            set(idx_text, 'HorizontalAlignment', 'center');
            set(idx_text, 'fontsize', 12);
            set(idx_text, 'FontWeight', 'bold');
        end

    end

    % reset flag
    inside_box_flag = 0;
    guidata(hObject, handles);
end


% --------------------------------------------------------------------
function Display_BOX_ID_Callback(hObject, eventdata, handles)
% hObject    handle to Display_BOX_ID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
checked = get(hObject, 'Checked');

    if strcmp(checked, 'off')
        set(hObject, 'Checked', 'on');
        handles.Show_mean_value_in_box.Checked = 'off';
        handles.ShowMeanVal=0;
        handles.DispBoxID=1;
        handles.previous_slice = handles.current_slice;
        handles.previous_volume = handles.current_volume;
        handles.hBoxes=Refresh_One_Frame(hObject, handles);
    else
        set(hObject, 'Checked', 'off');
        handles.DispBoxID=0;
        handles.previous_slice = handles.current_slice;
        handles.previous_volume = handles.current_volume;
        handles.hBoxes=Refresh_One_Frame(hObject, handles);
    end
    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function Show_mean_value_in_box_Callback(hObject, eventdata, handles)
% hObject    handle to Display_BOX_ID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
checked = get(hObject, 'Checked');

if strcmp(checked, 'off')
    set(hObject, 'Checked', 'on');
    handles.Display_BOX_ID.Checked = 'off';
    handles.DispBoxID=0;
    handles.ShowMeanVal=1;
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
else
    set(hObject, 'Checked', 'off');
    handles.ShowMeanVal=0;
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
end
guidata(hObject, handles);
end
    
    % --------------------------------------------------------------------
function colormap_jet_Callback(hObject, eventdata, handles)
    % hObject    handle to colormap_jet (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
checked = get(hObject, 'Checked');
if strcmp(checked, 'off')
    set(hObject, 'Checked', 'on');
    handles.CmapJet = 1;
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
else
    set(hObject, 'Checked', 'off');
    handles.CmapJet = 0;
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
end
guidata(hObject, handles);
end

%---------------------------------------------------------------------

% --------------------------------------------------------------------
function show_minimap_Callback(hObject, eventdata, handles)
% hObject    handle to show_minimap (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
checked = get(hObject, 'Checked');
if strcmp(checked, 'off')
    set(hObject, 'Checked', 'on');
    handles.ShowMinimap = 1;
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
else
    set(hObject, 'Checked', 'off');
    handles.ShowMinimap = 0;
    handles.previous_slice = handles.current_slice;
    handles.previous_volume = handles.current_volume;
    handles.hBoxes=Refresh_One_Frame(hObject, handles);
end
guidata(hObject, handles);
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double
    handles.neuron2find=str2num(get(hObject,'String'));
    if ~isempty(handles.neuron2find)
        vol=handles.current_volume;
        sli=handles.current_slice;
        j=1;
        handles.boxLocation=[];
        for s=1:handles.slices
            for i=1:length(handles.Boxes{vol,s})
                if ismember(handles.Boxes{vol,s}(i).idx,handles.neuron2find)
                    handles.boxLocation(j) = s;
                    j=j+1;
                end
            end
        end
        set(handles.neuronText,'String',['neurons: ',num2str(handles.neuron2find),...
            ' are on ',num2str(handles.boxLocation), ' slices']);
    else
        set(handles.neuronText,'String','');
    end
    guidata(hObject, handles);
end


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end
