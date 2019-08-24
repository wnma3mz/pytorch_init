## 一些Tips

### 关于TF与PyTorch修改梯度

```python
# tensorflow
grads_and_vars = opt.compute_gradients(loss, parameter_list)
my_grads_and_vars = [(g*C, v) for g, v in grads_and_vars]
opt.apply_gradients(my_grads_and_vars)

# pytorch
loss.backward()
for p in model.parameters():
    p.grad *= C  # or whatever other operation
optimizer.step()

# 如果有两个loss，修改其中一个
# retain_graph=True, 确保loss传播完保留。默认为False，为节省显存，在反向传播完之后会清空变量求导
loss2.backward(retain_graph=True) 
for p in model.parameters():
    p.grad *= C
optimizer.step()
optimizer.zero_grad()
loss1.backward()
optimizer.step()
```

```python3
# 梯度修改
weights = model.parameters()
loss.backward()
for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
```

### 梯度下降算法，pytorch源码剖析

https://www.rogoso.info/optim-method/
