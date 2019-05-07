
addpath('external\matlabPyrTools','external\randomforest-matlab\RF_Reg_C');

im=imread('1.jpg');

score=quality_predict(im);