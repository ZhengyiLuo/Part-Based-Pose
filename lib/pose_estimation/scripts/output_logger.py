from progress.bar import Bar

def matrix_loss_logger(current_epoch, epochs, steps_per_epoch):
    bar = Bar(
        "Epoch %d/%d" % (current_epoch, epochs),
        suffix=("%(index)d/%(max)d - loss: %(loss).7f - "
                ""),
        max=steps_per_epoch,
        loss=0.0,
    )
    bar.hide_cursor = False
    return bar

def get_logger(loss_type, current_epoch, epochs, steps_per_epoch):
    return {
        "matrix_loss": matrix_loss_logger
    }[loss_type](current_epoch, epochs, steps_per_epoch)
