function loss = log_cosh_loss(yPred, yTrue, mbs)
    
    loss = 1/mbs*(sum(log(cosh(yTrue - yPred))));

end
