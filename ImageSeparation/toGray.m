function toGray()
%TOGRAY �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
img=imadjust(dcm(i).img,[0.5;0.55],[0;1]); %�����Ҷȷ�Χ2   
img=double(img);              %���Ҷȼ�ӳ�䵽0~2553    
low=min(min(img));
high=max(max(img));
maxgray=high-low;
%���㴰��6      
rate=256/maxgray;
img=dcm(i).img*rate;
img=img+abs(min(min(img)));    %�Ӵ�9  
img=uint8(img);           %ת��Ϊ8λ��λͼ���ݸ�ʽ

end

