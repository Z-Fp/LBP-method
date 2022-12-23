%1399/08/18
%using lbp features to detect real and fake fingerprint images
%120 sample for train and 400 sample for test
%zahra farzadpour
close all
clear all
clc
%read input images(for train)
imagepath5='train';
filelist5=dir(fullfile(imagepath5,'*.bmp'));
list5={filelist5.name};
for i=1:length(list5)
    img5{i,1}=imresize(imread(fullfile(imagepath5,list5{i})),[96 96]); 
    
end
data_train=[img5];
for i=1:120
    input1=data_train{i,1};
for row=2:95
    for col=2:95
        centerpixel=input1(row,col);
        pixel7=input1(row-1,col-1)>centerpixel;
        pixel6=input1(row-1,col)>centerpixel;
        pixel5=input1(row-1,col+1)>centerpixel;
        pixel4=input1(row,col+1)>centerpixel;
        pixel3=input1(row+1,col+1)>centerpixel;
        pixel2=input1(row+1,col)>centerpixel;
        pixel1=input1(row+1,col-1)>centerpixel;
        pixel0=input1(row,col-1)>centerpixel;
        lbp_image(row,col)=uint8(pixel7*2^7+pixel6*2^6+pixel5*2^5+pixel4*2^4+pixel3*2^3+pixel2*2^2+pixel1*2^1+pixel0);
        x{i,1}=lbp_image;
    end
end
end
xx=[];
for i=1:120
    xx=[xx,reshape(x{i,1},9025,1)];
end
xdata=double([xx]');
%label
for q=1:60
    group{q,1}='real';
end
for q=61:120
    group{q,1}='fake';
end
%svm struct
svmStruct= svmtrain(xdata,group,'kernel_function','rbf','rbf_sigma',300,'showplot',false);
%testing 
%read input images
%input images for testing
imagepath1='test';
filelist1=dir(fullfile(imagepath1,'*.bmp'));
list1={filelist1.name};
for i=1:length(list1)
    img1{i,1}=imresize(imread(fullfile(imagepath1,list1{i})),[96 96]); 
    
end
data_test=[img1];
for i=1:400
    input11=data_test{i,1};
for row=2:95
    for col=2:95
        centerpixel=input11(row,col);
        pixel7=input11(row-1,col-1)>centerpixel;
        pixel6=input11(row-1,col)>centerpixel;
        pixel5=input11(row-1,col+1)>centerpixel;
        pixel4=input11(row,col+1)>centerpixel;
        pixel3=input11(row+1,col+1)>centerpixel;
        pixel2=input11(row+1,col)>centerpixel;
        pixel1=input11(row+1,col-1)>centerpixel;
        pixel0=input11(row,col-1)>centerpixel;     
        lbp_image_test(row,col)=uint8(pixel7*2^7+pixel6*2^6+pixel5*2^5+pixel4*2^4+pixel3*2^3+pixel2*2^2+pixel1*2^1+pixel0);
        s{i,1}=lbp_image_test;
    end
end
end
 ss=[];
 for i=1:400
    ss=[ss,reshape(s{i,1},9025,1)];
end
sample=double([ss]');
%svm test
Test = svmclassify(svmStruct,sample,'showplot',false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%f-score
for i=1:300
    actual{i,1}=[0];
end
for i=301:400
    actual{i,1}=[1];
end
for i=1:400
    if Test{i,1}=='real'
        predicted{i,1}=[0];
    else
        predicted{i,1}=[1];
    end
end
ACTUAL=(cell2mat(actual));
PREDICTED=(cell2mat(predicted));
result=fscore(ACTUAL,PREDICTED);


