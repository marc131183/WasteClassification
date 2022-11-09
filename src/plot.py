import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def lossAccuracyPlot():
    all_loss = [
        1.2810353729034256,
        0.8130112157651505,
        0.5588402834760255,
        0.42800147937929284,
        0.34440539852702945,
        0.26082779066319,
        0.23048359790053696,
        0.1383889604458984,
        0.11490156943481532,
        0.10257709576305524,
        0.08546275927884568,
        0.0739588669102322,
        0.07476941123630661,
        0.0621119377435128,
        0.05208279959991034,
        0.05748265185249374,
        0.07218870618551687,
    ]
    all_acc = [
        0.5149,
        0.7115,
        0.8078,
        0.8541,
        0.8880,
        0.9163,
        0.9252,
        0.9596,
        0.9693,
        0.9717,
        0.9766,
        0.9808,
        0.9778,
        0.9834,
        0.9887,
        0.9844,
        0.9818,
    ]
    subset_loss = [
        4.778817755904624,
        3.331614025118874,
        2.6053166224098785,
        1.8822620018892657,
        1.327087240127044,
        1.0818040079732494,
        0.7922067567478956,
        0.6129448067934472,
        0.5234881673553368,
        0.3282281436116957,
        0.3361929923795709,
        0.29060911979314275,
        0.40655819831294865,
        0.3346151734505424,
        0.3170553507430979,
        0.23922868122013544,
        0.18414817537499092,
    ]

    fig, ax = plt.subplots()

    ax.plot(all_loss, label="Loss on all data", color="#7fc97f")
    ax.plot(subset_loss, label="Loss on subset", color="#beaed4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.grid(linestyle="--")
    ax.yaxis.grid(linestyle="--")

    y_lim = (0, 5)
    n = 11

    ax.set_yticks(np.arange(0, 5.5, 0.5))
    ax.set_ylim(*y_lim)
    ax.set_yticks(np.linspace(*y_lim, n))

    y_lim = (0.5, 1)

    ax2 = ax.twinx()
    ax2.plot(all_acc, label="Accuracy on all data", color="#fdc086")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(*y_lim)
    ax2.set_yticks(np.linspace(*y_lim, n))

    fig.legend(loc="center")

    plt.xticks(np.arange(len(all_acc)))

    plt.show()


def accuracyComparisonPlot():
    # fmt: off
    alexnet_accuracy_traditional = [0.5204918032786885, 0.6101010101010101, 0.6310483870967742, 0.6544715447154471, 0.784, 0.7626774847870182, 0.725609756097561, 0.6673387096774194, 0.6707070707070707, 0.75]
    alexnet_accuracy_other = [0.5266393442622951, 0.6282828282828283, 0.6048387096774194, 0.6646341463414634, 0.75, 0.7809330628803245, 0.7479674796747967, 0.6229838709677419, 0.6121212121212121, 0.71]

    resnet18_accuracy_traditional = [0.8155737704918032, 0.8626262626262626, 0.9435483870967742, 0.8922764227642277, 0.96, 0.8762677484787018, 0.9390243902439024, 0.8528225806451613, 0.795959595959596, 0.848]
    resnet18_accuracy_other = [0.5942622950819673, 0.7414141414141414, 0.8931451612903226, 0.8130081300813008, 0.962, 0.8032454361054767, 0.9247967479674797, 0.7741935483870968, 0.6747474747474748, 0.706]

    resnet50_accuracy_traditional = [0.8258196721311475, 0.8828282828282829, 0.9112903225806451, 0.8963414634146342, 0.956, 0.8823529411764706, 0.9776422764227642, 0.8165322580645161, 0.9191919191919192, 0.85]
    resnet50_accuracy_other = [0.6577868852459017, 0.896969696969697, 0.8528225806451613, 0.8577235772357723, 0.93, 0.821501014198783, 0.9065040650406504, 0.8104838709677419, 0.8525252525252526, 0.798]

    vgg_accuracy_traditional = [0.694672131147541, 0.8525252525252526, 0.8487903225806451, 0.806910569105691, 0.918, 0.8640973630831643, 0.8841463414634146, 0.7399193548387096, 0.7515151515151515, 0.838]
    vgg_accuracy_other = [0.6352459016393442, 0.8363636363636363, 0.8044354838709677, 0.8109756097560976, 0.908, 0.8397565922920892, 0.8841463414634146, 0.7459677419354839, 0.7313131313131314, 0.81]
    # fmt: on
    ticks = ["AlexNet", "ResNet18", "ResNet50", "VGG"]

    traditional_accuracies = [
        alexnet_accuracy_traditional,
        resnet18_accuracy_traditional,
        resnet50_accuracy_traditional,
        vgg_accuracy_traditional,
    ]
    other_accuracies = [
        alexnet_accuracy_other,
        resnet18_accuracy_other,
        resnet50_accuracy_other,
        vgg_accuracy_other,
    ]

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)

    fig = plt.figure()

    bpl = plt.boxplot(
        traditional_accuracies,
        positions=np.array(range(len(traditional_accuracies))) * 2.0 - 0.4,
        sym="",
        widths=0.6,
    )
    bpr = plt.boxplot(
        other_accuracies,
        positions=np.array(range(len(other_accuracies))) * 2.0 + 0.4,
        sym="",
        widths=0.6,
    )
    set_box_color(bpl, "#D7191C")  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, "#2C7BB6")

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="#D7191C", label="Traditional Approach")
    plt.plot([], c="#2C7BB6", label="Our Approach")
    plt.legend(loc="lower right")

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.xlim(-1.2, len(ticks) * 2 - 0.8)
    plt.ylim(0.5, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model Architecture")
    plt.gca().yaxis.grid(True, linestyle="--")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    accuracyComparisonPlot()
