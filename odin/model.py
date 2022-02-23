
class Odin_model:

	def __init__(self):



	def train_model(self, num_epochs: int=100):
		loss_hist = Averager()
		itr = 1

		for epoch in range(num_epochs):
		    loss_hist.reset()
		    
		    for images, targets, image_ids in train_data_loader:
		        images = list(image.to(device) for image in images)
		        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		        for index in range(0,len(targets)):
		            targets[index]['labels']=torch.ones(len(targets[index]['boxes']),dtype=torch.int64).to(device)
		        loss_dict = model(images, targets)

		        losses = sum(loss for loss in loss_dict.values())
		        loss_value = losses.item()

		        loss_hist.send(loss_value)

		        optimizer.zero_grad()
		        losses.backward()
		        optimizer.step()

		        if itr % 50 == 0:
		            print(f"Iteration #{itr} loss: {loss_value}")

		        itr += 1
		    
		    checkpoint = {
		            'epoch': epoch + 1,
		            'train_loss_min': loss_hist.value,
		            'state_dict': model.state_dict(),
		            'optimizer': optimizer.state_dict(),
		        }
		    
		    save_ckp(checkpoint, False, checkpoint_path, best_model_path)
		      
		    ## save the model if validation loss has decreased
		    if loss_hist.value <= train_loss_min:
		        print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,loss_hist.value))
		        # save checkpoint as best model
		        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
		        train_loss_min = loss_hist.value
		    
		    # update the learning rate
		    if lr_scheduler is not None:
		        lr_scheduler.step()

		    print(f"Epoch #{epoch} loss: {loss_hist.value}")  




class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 30
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

