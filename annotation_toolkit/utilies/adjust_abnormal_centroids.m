function centroids = adjust_abnormal_centroids( centroids,x_max,y_max )

N=size(centroids,2);

for i=1:N
    if (centroids(1,i)<=1) 
        centroids(1,i)=10;
    end
    
    if (centroids(1,i)>x_max) 
        centroids(1,i)=x_max-10;
    end
    
    if (centroids(2,i)<=1) 
        centroids(2,i)=10;
    end
    
    if (centroids(2,i)>y_max) 
        centroids(2,i)=y_max-10;
    end
    
    

end

