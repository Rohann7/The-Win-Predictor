function op=pro_result(inp,Theta1,Theta2);





inp=[ones(1,1) inp];

act=(inp*(Theta1)');
act1=[ones(1,1) act];

for i=1:(size(act,2)),
act(:,i)=pro_sigmoid(act(:,i));
end

act=[ones(1,1) act];

out=(act*(Theta2)');


out=pro_sigmoid(out);

op=out*100;



end
 


