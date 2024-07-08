function iter_plot_T2(epoch,plot_freq,loss,y_true,y_pred, y_upper, y_lower)
persistent hasRun
persistent start_time
persistent line_loss_train_persistent
persistent plotting_timesteps

if epoch == 1
    hasRun = false;
    start_time = 0;
    line_loss_train_persistent = 0;
    plotting_timesteps = 0;

end

if hasRun == false

    start_time = tic;
    hasRun = true;

    f = figure;
    f.Position(2) = 2*f.Position(2);
    subplot(1, 2, 1)
    C = colororder;
    line_loss_train_persistent = animatedline(Color=C(2,:));
    ylim([0 inf])
    xlabel("Epoch")
    ylabel("Loss")
    grid on


    plotting_timesteps = max(size(y_true));
    plotting_timesteps = 1:plotting_timesteps;

    
end

 %Plot Loss
    subplot(1, 2, 1)
    current_loss = double(loss);

    line_loss_train = line_loss_train_persistent;
%     line_loss_train = animatedline(Color=C(2,:));

    addpoints(line_loss_train,epoch,current_loss);
    D = duration(0,0,toc(start_time),Format="hh:mm:ss");
    title("Elapsed: " + string(D))
    drawnow

     if mod(epoch,plot_freq) == 0  || epoch == 1

         subplot(1, 2, 2)

         plot(plotting_timesteps, y_true , "rx");
         hold on
         plot(plotting_timesteps, y_pred, "b");
         plot(plotting_timesteps, y_upper, "y");
%          plot(plotting_timesteps, y_lower);
         hold off

     end

end