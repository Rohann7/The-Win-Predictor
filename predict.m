function [h2,h4,h6,JC,JT,J] = predict(Theta1, Theta2, X,Y,XC,YC,XT,YT,lambda)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 


h1 = pro_sigmoid([ones(m, 1) X] * Theta1');
h2 = pro_sigmoid([ones(m, 1) h1] * Theta2');

m = size(XC, 1);

h3 =  pro_sigmoid([ones(m, 1) XC] * Theta1');
h4 = pro_sigmoid([ones(m, 1) h3] * Theta2');

m1= size(XC, 1);

h5 =  pro_sigmoid([ones(m, 1) XT] * Theta1');
h6 = pro_sigmoid([ones(m, 1) h5] * Theta2');

for i=1:(length(h2)),
   if h2(i)>=0.5,
   h2(i)=1;
   else
   h2(i)=0;
   end;
   
   end;
   
   for i=1:(length(h4)),
   if h4(i)>=0.5,
   h4(i)=1;
   else
   h4(i)=0;
   end;
   
   end;
   
   for i=1:(length(h6)),
   if h6(i)>=0.5,
   h6(i)=1;
   else
   h6(i)=0;
   end;
   
   end;




n =  size(Theta1,2);
p =  size(Theta2,2);  

XC=[ones(m,1) XC];

act=(XC*(Theta1)');
act1=[ones(m,1) act];

for i=1:(size(act,2)),
act(:,i)=pro_sigmoid(act(:,i));
end

act=[ones(m,1) act];

out=(act*(Theta2)');


for i=1:(size(out,2)),

out(:,i)=pro_sigmoid(out(:,i));
 
end

out1=log(out);
out2=log(1-out);


JC=(sum(( -YC.*(out1))-(( 1-YC ).*out2)))/(m);

XT=[ones(m,1) XT];

act=(XT*(Theta1)');
act1=[ones(m,1) act];

for i=1:(size(act,2)),
act(:,i)=pro_sigmoid(act(:,i));
end

act=[ones(m,1) act];

out=(act*(Theta2)');


for i=1:(size(out,2)),

out(:,i)=pro_sigmoid(out(:,i));
 
end

out1=log(out);
out2=log(1-out);


JT=(sum(( -YT.*(out1))-(( 1-YT ).*out2)))/(m1);




m= size(X, 1);

n =  size(Theta1,2);
p =  size(Theta2,2);  

X=[ones(m,1) X];

act=(X*(Theta1)');
act1=[ones(m,1) act];

for i=1:(size(act,2)),
act(:,i)=pro_sigmoid(act(:,i));
end

act=[ones(m,1) act];

out=(act*(Theta2)');


for i=1:(size(out,2)),

out(:,i)=pro_sigmoid(out(:,i));
 
end

out1=log(out);
out2=log(1-out);


J=(sum(( -Y.*(out1))-(( 1-Y ).*out2)))/(m)+(((sum(sum((Theta1(:,(2:n)).^2),2)))+sum(sum((Theta2(:,(2:p)).^2),2)))*(lambda/(2*m)));

% =========================================================================


end
