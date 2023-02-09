## Nan issue with large training dataset, large layer etc.. 04 Feb 2023
this error started when I tried to classify 3 cluster data set I generated with large not normalized inputs and a lot of samples,
at the beggining the small network I worked with didn't manage to classify 3 clusters (when creating a 2 cluster dataset it worked)
when I increased the numebr of samples and add neurons I started getting NaN values.
after a lot of debugging I found that the values that are inserted to the softmax function at the end of the network are around [-800, -800, -800] which then led to NaN values.
these values are too big considering the input values, the expected outputs and the learning rate.
I identified this issue as exploding gradients in combination with non normalized input data.
normalizing the input data helped but to consider network solutions
1) change the weight initialization method 
2) consider gradient clipiing
