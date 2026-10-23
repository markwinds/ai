# 无框架训练一个线性模型
# y=wx+b

xs = [1.0, 2.0, 3.0, 4.0]
ys = [3.0, 5.0, 7.0, 9.0]

w = 0.0
b = 0.0
lr = 0.01
n = len(xs)

for epoch in range(2000):
    dw = 0.0
    db = 0.0
    loss = 0.0

    # 前向传播+计算梯度
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y

        loss += error ** 2
        dw += 2 * error * x
        db += 2 * error

    loss /= n
    dw /= n
    db /= n

    # 梯度下降
    w -= lr * dw
    b -= lr * db

    if epoch % 200 == 0:
        print(f"epoch={epoch:4d}, "
              f"loss={loss:.6f}, "
              f"w={w:.4f}, b={b:.4f}")

print("\n最终参数：")
print(f"w = {w:.6f}")
print(f"b = {b:.6f}")

# 验证学习结果
assert abs(w - 2.0) < 0.05
assert abs(b - 1.0) < 0.05

print("PASS: 模型成功学习线性关系")
