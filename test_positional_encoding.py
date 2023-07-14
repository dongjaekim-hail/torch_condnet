import torch
from torch import nn
from matplotlib import pyplot as plt
class PositionalEncoding(nn.Module):
	def __init__(self,device,period=10000):
		super().__init__()
		self.device = device
		self.period = period

	def forward(self,x):
		self.P_E = x
		# 학습되는 값이 아님으로 requires_grad 을 False로 설정
		self.P_E.requires_grad = False
		# x seq 길이에 맞춰 PE return
		# (seq_len, d_model)

		max_len = self.P_E.size(0)
		d_model = self.P_E.size(1)


		# 2. pos (0~max_len) 생성 (row 방향 => unsqueeze(dim=1))
		pos = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(dim=1)

		# 3. _2i (0~2i) 생성 (col 방향)
		# 2i는 step = 2 를 활용하여 i의 2배수를 만듦
		_2i = torch.arange(0, d_model, step= 2, dtype=torch.float, device=self.device)

		# 4. 제안된 positional encoding 생성
		# (i가 짝수일때 : sin, 홀수일때 : cos)
		self.P_E[:, 0::2] = torch.sin(pos / self.period ** (_2i / d_model))
		self.P_E[:, 1::2] = torch.cos(pos / self.period ** (_2i / d_model))

		return self.P_E


if __name__ == "__main__":
	# test Positional Encoding class


	#list of periods
	periods = [10000, 2, 1, 0.5, 0.01]
	# subplots
	fig, axes = plt.subplots(len(periods), 1, figsize=(15, 5))

	for i in range(len(periods)):
		# 1. Positional Encoding 초기화
		PE = PositionalEncoding(device='cpu', period=periods[i])
		# 1.1. 비어있는 tensor 생성
		# (max_len,d_model)
		vector = torch.zeros(7, 1024, device='cpu')
		# 1.2.PE class를 사용해서 계산
		PE_for_vector = PE(vector)

		# plt 를 이용해서 PE_for_vector 를 heatmap 같은 형태로 플랏하기 (but as square shape)
		axes[i].imshow(PE_for_vector, cmap='viridis',aspect='auto')
		axes[i].set_title(f'period : {periods[i]}')
		axes[i].set_xlabel('d_model')
		axes[i].set_ylabel('max_len')
	plt.show()


	print('')

