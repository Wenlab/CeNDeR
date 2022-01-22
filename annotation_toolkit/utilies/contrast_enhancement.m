function img_enhanced = contrast_enhancement(img)

if ndims(img)==2
    se = strel('disk', 10);
    Itop = imtophat(img, se);
    Ibot = imbothat(img, se);
    img_enhanced = imsubtract(imadd(Itop, img), Ibot);

elseif ndims(img)==3
    M=size(img,3);
    img_enhanced=zeros(size(img));
    
    for j=1:M
         se = strel('disk', 10);
         Itop = imtophat(img(:,:,j), se);
         Ibot = imbothat(img(:,:,j), se);
         img_enhanced(:,:,j)=imsubtract(imadd(Itop, img(:,:,j)), Ibot);
    end
    
end


end

