import torch
from models import Generator, Discriminator

def weights_init_normal(m):
    # Initialize model
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def Models():
    #Create model 
    latent_dim = 4000
    g_net=Generator(latent_dim)
    d_net=Discriminator()
    g_optimizer = optim.Adam(g_net.parameters(), lr=0.0002, betas=(0.5,0.999), weight_decay=5e-4)
    d_optimizer = optim.Adam(d_net.parameters(), lr=0.0002, betas=(0.5,0.999), weight_decay=5e-4)
    # Move models
    g_net = g_net.to(dev)
    d_net = d_net.to(dev)
    return g_net,d_net,g_optimizer,d_optimizer

def compare_and_save(photo_to_save,path,test_loader,g_net):
    test_image= next(iter(test_loader))
    test_image = test_image.to(dev)
    test_masked_imgs =test_image.clone() 
    if not os.path.exists(path):
        os.makedirs(path)
    with torch.no_grad():
                # Removing center from the test sample
                sample = apply_center_mask(test_image)
                # Forward (generator)
                g_sample = g_net(sample)
                # Impanting the image generated to the original
                test_masked_imgs[:,:,(mask_size//2):img_size-(mask_size//2),(mask_size//2):img_size-(mask_size//2)] = g_sample.data
                for i in range(photo_to_save):
                    generated =  (test_image[i,:,:,:]).unsqueeze(0)
                    original =  (test_masked_imgs[i,:,:,:]).unsqueeze(0)
                    compare = torch.cat((original, generated), 0).clone()
                    save_image(compare, path+"%d.png" % i, nrow=2, normalize=True)   

def training(train_loader,test_loader,labels_noise=False,wtl2= 0.999,last_epoch=200,save_photos_interval=10,overlapL2Weight=10):
    '''
    train_loader: Dataloader of train data
    test_loader : Dataloader of test data
    labels_noise: Boolean that enable labels smoothing and flipping
    wtl2: param to weights losses 
    last_epoch: number of last epoch
    save_photos_interval: set interval of every x epoch generate photos to compare
    overlapL2Weight: weights amplified 

    '''
    # Define labels 
    valid = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)
    path_toSave_photos = "/content/images/"
    total_time = 0
    pattern = generate_pattern()
    # load test image
    test_image= next(iter(test_loader))
    test_image = test_image.to(dev)
    test_masked_imgs =test_image.clone() 
    # Create the models 
    g_net,d_net,g_optimizer,d_optimizer = Models()
    # If backup it's available load it 
    if os.path.isfile(save_path_discriminator) and os.path.isfile(save_path_generator) and restore:
        checkpoint_d =  torch.load(save_path_discriminator)
        checkpoint_g =  torch.load(save_path_generator)
        d_net.load_state_dict(checkpoint_d['d_state_dict'])
        d_optimizer.load_state_dict(checkpoint_d['optimizer_state_dict'])
        d_loss = checkpoint_d['loss']
        d_loss_fake = checkpoint_d['loss_fake']
        d_loss_real = checkpoint_d['loss_real']
        g_net.load_state_dict(checkpoint_g['g_state_dict'])
        g_optimizer.load_state_dict(checkpoint_g['optimizer_state_dict'])
        g_loss = checkpoint_g['loss']    
        g_loss_pixel = checkpoint_g['loss_pixel']
        g_loss_adv = checkpoint_g['loss_adv']
        epoch_backup = checkpoint_g['epoch']+1
        print("Discriminator and Generator restored")
    else :
        epoch_backup = 0 
        g_net.apply(weights_init_normal)
        d_net.apply(weights_init_normal)
        print("weight applied")
    try:
        for epoch in range(epoch_backup,last_epoch):
            # Losses
            start = time.time()
            sum_d_loss = 0
            sum_d_fake_loss = 0
            sum_d_real_loss = 0
            sum_g_loss = 0
            sum_g_loss_adv = 0
            sum_g_loss_pixel = 0
            # Training mode
            d_net.train()
            g_net.train()
            # Process all training batches
            i = 0
    
            for batch in train_loader:
                batch = batch.to(dev)
                # Move to device
                i+=1
                masked_parts = get_center(batch)
                #masked_parts are the center of the images 
                masked_parts = Variable(masked_parts.type(Tensor))
                masked_imgs = choise_mask(batch,pattern)
                masked_imgs = Variable(masked_imgs.type(Tensor))

                if labels_noise:
                    # probability of flipping 0.05
                    valid_flipped = noisy_labels(valid.clone(), 0.05)
                    fake_flipped = noisy_labels(fake.clone(), 0.05)  
                    valid_smooth = label_smoothing_valid(valid_flipped)
                    fake_smooth = label_smoothing_fake(fake_flipped)   

                ### Discriminator 
                # Reset discriminator gradient
                d_optimizer.zero_grad()
                # Forward (discriminator, real)
                output = d_net(masked_parts) 
                # Compute loss (discriminator, real)
                if labels_noise:
                  d_real_loss =  F.binary_cross_entropy(output, valid_smooth)
                else:
                  d_real_loss =  F.binary_cross_entropy(output, valid)
                # Backward (discriminator, real)
                d_real_loss.backward()
                sum_d_real_loss += d_real_loss.item()  
                #generate sample from masked images         
                g_output = g_net(masked_imgs)
                # Forward (discriminator, fake; also generator forward pass)
                output = d_net(g_output.detach()) # This prevents backpropagation from going inside the generator
                # Compute loss (discriminator, fake)
                if labels_noise:
                  d_fake_loss = F.binary_cross_entropy(output, fake_smooth)
                else:  
                  d_fake_loss = F.binary_cross_entropy(output, fake)
                # Backward (discriminator, fake)
                d_fake_loss.backward()
                sum_d_fake_loss += d_fake_loss.item()           
                d_loss = 0.5*(d_fake_loss + d_real_loss)
                sum_d_loss += d_loss.item()
                # Update discriminator
                d_optimizer.step()
                ### Generator 
                g_optimizer.zero_grad()
                # Forward (generator)
                output =  d_net(g_output)
                # Compute adversarial loss
                g_loss_adv = F.binary_cross_entropy(output, valid)            
                # Comput pixelwise loss
                # but amplifying weights 10x 
                #g_loss_pixel =  criterionMSE(g_output,masked_parts)
                wtl2Matrix = masked_parts.clone()
                # OverlapL2weight = 10
                wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
                wtl2Matrix.data[:,:,overlapPred:mask_size-overlapPred,overlapPred:mask_size-overlapPred] = wtl2
                # MSE Loss
                g_loss_pixel = (g_output-masked_parts).pow(2)
                # Multiply 
                g_loss_pixel = g_loss_pixel * wtl2Matrix
                g_loss_pixel = g_loss_pixel.mean()
                # The losse it's the sum of adv and pixel
                g_loss = (1-wtl2) * g_loss_adv + wtl2 * g_loss_pixel
                sum_g_loss_adv += g_loss_adv.item()
                sum_g_loss_pixel += g_loss_pixel.item()
                sum_g_loss += g_loss.item()
                # Backward (generator)
                g_loss.backward()
                # Update generator
                g_optimizer.step()
                if (i%700==0):
                    print(f"Batches {i}/{len(train_loader)}")

            # Epoch end, print losses
            epoch_d_loss = sum_d_loss/len(train_loader)
            epoch_d_real_loss = sum_d_real_loss/len(train_loader)
            epoch_d_fake_loss = sum_d_fake_loss/len(train_loader)
            epoch_g_loss_adv = sum_g_loss_adv/len(train_loader)
            epoch_g_loss_pixel = sum_g_loss_pixel/len(train_loader)
            epoch_g_loss = sum_g_loss/len(train_loader)
            end = time.time()   
            time_epoch = (end - start)/60
            total_time +=time_epoch
            # Save models
            torch.save({'g_state_dict': g_net.state_dict(),
                        'optimizer_state_dict': g_optimizer.state_dict(),
                        'loss': g_loss,
                        'loss_pixel': g_loss_pixel,
                        'loss_adv': g_loss_adv,
                        'epoch':epoch,
                        }, save_path_generator)

            torch.save({'d_state_dict': d_net.state_dict(),
                        'optimizer_state_dict': d_optimizer.state_dict(),
                        'loss_fake': d_fake_loss,
                        'loss_real': d_real_loss,
                        'loss': d_loss,
                        'epoch': epoch
                        }, save_path_discriminator)
            if ((epoch+1)%save_photos_interval==0):
                compare_and_save(64,path_toSave_photos,test_loader,g_net)
            print(f"Epoch {epoch+1} DL={epoch_d_loss:.4f} DR={epoch_d_real_loss:.4f} DF={epoch_d_fake_loss:.4f} GL={epoch_g_loss:.4f} GLP={epoch_g_loss_pixel:.4f} GLADV={epoch_g_loss_adv:.4f} Time {time_epoch:.1f}min Total Time: {total_time/60 :.1f}h")
            # Evaluation mode
            g_net.eval()
            with torch.no_grad():
                # Removing center from the test sample
                sample = apply_center_mask(test_image)
                # Forward (generator)
                g_sample = g_net(sample)
                # Impanting the image generated to the original
                test_masked_imgs[:,:,(mask_size//2):img_size-(mask_size//2),(mask_size//2):img_size-(mask_size//2)] = g_sample.data
                plt.imshow(TF.to_pil_image(make_grid(test_masked_imgs[:4], scale_each=True, normalize=True).cpu()))
                plt.axis('off')
                plt.show()
                
    except KeyboardInterrupt:
          print("Interrupted")            
     