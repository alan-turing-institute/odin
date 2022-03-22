import torch
import torchvision
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from odin.io import save_checkpoint, show_each_image, g_to_rgb


class Odin_model:

    def __init__(
            self,
            checkpoint_path: str = "models/checkpoint.pt",
            device: str = "default",
            nms: float = 0.75,
            pretrained: bool = True,
            num_classes: int = 2):
        """
        Initialise the Odin model.

        """

        # Initialise the model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                          trainable_backbone_layers=2,
                                                                          progress=False,
                                                                          rpn_nms_thresh=nms
                                                                          )

        if device in ["default", None]:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device

        self.model.to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.optimizer = torch.optim.SGD(self.params, 0.005, 0.9, 0.0005)
        self.lr_scheduler = None
        if pretrained:
            self.load(checkpoint_path)
            print("Model loaded from " + checkpoint_path)

        else:
            self.checkpoint_path = '/models/nms' + str(nms) + '_chkpoint_'
            self.best_model_path = '/models/nms' + str(nms) + '_bestmodel.pt'
            print("New model: to train, use function .train()")
        self.writer = SummaryWriter()

    def set_optimizer(self, opt_type: str = "default", lr: float = 0.005, momentum: float = 0.9,
                      weight_decay: float = 0.0005, lr_scheduler=None):
        if opt_type in ["default", "SGD"]:
            self.optimizer = torch.optim.SGD(self.params, lr, momentum, weight_decay)
        self.lr_scheduler = lr_scheduler

    def load(self,
             checkpoint_path: str = "models/checkpoint.pt"):
        """
            checkpoint_path: path to save checkpoint
        """
        # load check point
        checkpoint = torch.load(checkpoint_path)

        # initialize state_dict from checkpoint to model
        self.model.load_state_dict(checkpoint['state_dict'])

        # initialize optimizer from checkpoint to optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, train_data_loader, num_epochs: int = 100, train_loss_min: float = 3):
        loss_hist = Averager()
        itr = 1

        for epoch in range(num_epochs):
            loss_hist.reset()

            for images, targets, image_ids in train_data_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                for index in range(0, len(targets)):
                    targets[index]['labels'] = torch.ones(len(targets[index]['boxes']), dtype=torch.int64).to(
                        self.device)
                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                loss_hist.send(loss_value)

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                if itr % 50 == 0:
                    print(f"Iteration #{itr} loss: {loss_value}")

                itr += 1

                # grid = torchvision.utils.make_grid(images)
                # self.writer.add_image('images', grid, 0)
                # self.writer.add_graph(self.model, images)
                self.writer.add_scalar('Loss/train', loss_value, itr)
                # self.writer.add_scalar('Accuracy/train', np.random.random(), itr)

            checkpoint = {
                'epoch': epoch + 1,
                'train_loss_min': loss_hist.value,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, False, self.checkpoint_path, self.best_model_path)

            # save the model if validation loss has decreased
            if loss_hist.value <= train_loss_min:
                print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,
                                                                                           loss_hist.value))
                # save checkpoint as best model
                save_checkpoint(checkpoint, True, self.checkpoint_path, self.best_model_path)
                train_loss_min = loss_hist.value

            # update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(f"Epoch #{epoch} loss: {loss_hist.value}")

    def predict(self, image, print_result: bool = False):
        """
        Receive the image and retrieves the predicted bounding boxes.
        """

        self.model.eval()
        if type(image).__module__ != np.__name__:
            image = image.to_numpy()
        if len(image) < 3:
            image = g_to_rgb(image)
        image = torch.from_numpy(image).float()
        output = self.model([image.to(self.device)])
        bboxes = output[0]['boxes']
        if print_result:
            show_each_image(image, bboxes)

        return bboxes


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
