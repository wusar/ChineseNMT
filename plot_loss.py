import matplotlib.pyplot as plt

# 提取的数据
epochs = [1, 2, 3, 4, 5]
loss = [7.5809502601623535, 6.109577178955078, 5.169185161590576, 4.581043243408203, 4.21594762802124]
dev_loss = [6.627676486968994, 5.513684272766113, 4.787280559539795, 4.433265686035156, 4.228055000305176]
bleu_score = [2.1521661297189594, 10.60102071349868, 17.343057759963667, 19.84035773952465, 21.733959882420745]

# 绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, marker='o', linestyle='-', color='r', label='Loss')
plt.plot(epochs, dev_loss, marker='o', linestyle='-', color='b', label='Dev Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# 绘制bleu score曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, bleu_score, marker='o', linestyle='-', color='g', label='Bleu Score')
plt.xlabel('Epochs')
plt.ylabel('Bleu Score')
plt.legend()
plt.grid()
plt.show()