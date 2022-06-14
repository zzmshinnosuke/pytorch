# pytorch
### Tensor, Variable, Parameter的区别   
- Variable 在新版本已经弃用了，完全可以用Tensor代替。 
- Tensor 默认是不求梯度的，如果要求梯度需要手动设置`requires_grad=True`, nn.Parameter是Tensor的子类，不过Parameter默认是求梯度的，同时网络中的Parameter变量是可以通过net.parameters()来访问。使用torch的nn.Module时，即使初始化一些不需要计算梯度的量，也应该初始化为Parameter，因为model.to(device)是将Parameter移动到device上，但是并不会将tensor移动到device上。 
