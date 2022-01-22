function img_Stack=import_micromanager_data_and_reshape(varargin)

if exist('pathname', 'var')
    try
        if isdir(pathname)
            cd(pathname);
        end
    end
end
if nargin==0 || strcmp(varargin{1},'imageStack')
    [filename,pathname]  = uigetfile({'*.tif'});
    fname = [pathname filename];
    info = imfinfo(fname); 
    num_frames=length(info);
    img_stack=cell(num_frames,1);
    for j=1:num_frames
        img_stack{j,1}=imread(fname,j, 'Info', info); 
        if mod(j,100)==0
            disp(j);
        end
        
    end
elseif strcmp(varargin{1},'imageSequence')
    [pathname]=uigetdir();
    prompt = {'start frame:','end frame:'};
    dlgtitle = 'count from 1';
    dims = [1 35];
    definput = {'',''};
    answer = inputdlg(prompt,dlgtitle,dims,definput);    
    istart=str2num(answer{1});
    iend=str2num(answer{2});
    listing=dir(pathname);
    imageIdx=zeros(length(listing),1 );
    listingIdx=(1:length(listing))';
    %%
    for i=1:length(listing)
        if ~listing(i).isdir && strcmp(listing(i).name(end-3:end),'.tif')
           imageIdx(i)=str2num(listing(i).name(1:end-3));
           fname{i}=fullfile(listing(i).folder,listing(i).name);
        end
    end
    %%
    [sortedImageIdx,imageIdxListing]=sort(imageIdx);
    if iend>sortedImageIdx(end)
        fprintf(['WARNING! The maximum index of .tif image is %d,\n', ...
        'the input iend is %d, which exceeds it. Set iend to %d'],sortedImageIdx(end),iend,sortedImageIdx(end));
        iend=sortedImageIdx(end);
    elseif iend-istart+1>sum(imageIdx~=1)
        fprintf('WARNING! Number of images inputed (%d) exceeds valid range, setting iend to vaild number %d',...
        iend-istart+1,sortedImageIdx(end));
        iend=sortedImageIdx(end);
    end

    framesBehind=sortedImageIdx>=istart;
    framesAhead=sortedImageIdx<=iend;
    framesInside=framesBehind+framesAhead;
    listingInside=imageIdxListing(framesInside==2);
    num_frames=iend-istart+1;
    for j=1:num_frames
        info=imfinfo(fname{listingInside(j)});
        img_stack{j,1}=imread(fname{listingInside(j)},'info',info);
         if mod(j,100)==0
            disp(j);
        end
    end
end
answer = inputdlg({'Start frame (exclude not in complete cycle frames)'...
    , 'End frame (exclude not in complete cycle frames)','number of z slices per volume'}, '', 1);
istart = str2double(answer{1});
iend = str2double(answer{2});
if iend>length(img_stack)
    iend=length(img_stack);
    disp(iend);
end

num_z=str2double(answer{3});
num_t=floor((iend-istart+1)/num_z);
img_Stack=cell(num_t,1);

answer = inputdlg({'Start z slice to project (relative frame in one cycle)'...
    , 'End z slice to project (relative frame in one cycle)'}, '', 1);
start_z = str2double(answer{1});
end_z = str2double(answer{2});
len_z=end_z-start_z+1;

[n,m]=size(img_stack{1,1});

for k=1:num_t
    
    i=(k-1)*num_z+istart;
    
    img_stack_temp=zeros(n,m,len_z);
                    
    for j=1:len_z
        img_stack_temp(:,:,j)=img_stack{i+start_z+j-2,1};
    end
            
    img_Stack{k,1}=uint16(img_stack_temp);
    if mod(k,50)==0
            disp(k);
    end
    
end
fprintf("Whole stack: start frame  %d, end frame  %d.\n",istart,iend);
fprintf("Each volume: start slice  %d, end slice  %d.\n", start_z,end_z);
end
    









    




