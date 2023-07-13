gradients = []
for param in model.parameters():
    gradients.append(param.grad.data.view(-1))
gradients = torch.cat(gradients)


gradients = []
for param in model.parameters():
    param
