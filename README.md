﻿### Amazon-Reviews

Simple model to analyze sentiment of the amazon reviews.

Training took approximately 21 hours. Hyperparameters changing and fine-tuning model
would be very time demanding task. There were 20 epochs and below you can see the results:

![Accuracy image](img/Accuracy_2024-02-10.png)

**Accuracy on the test samples reached around 70%**

There is no characteristic in regard to test samples, training samples and their loss
because there would need a lot more epochs to train this model on. That means a lot more
time training.

Below you can find example of the data structure
<details>
<summary>Data sample</summary>

```text
__label__2 Remember, Pull Your Jaw Off The Floor After Hearing it:
           If you've played the game, you know how divine the music is!
           Every single song tells a story of the game, it's that good!
           The greatest songs are without a doubt, Chrono Cross:
           Time's Scar, Magical Dreamers: The Wind, The Stars,
           and the Sea and Radical Dreamers: Unstolen Jewel. 
           (Translation varies) This music is perfect if you ask me, the best it can be.
           Yasunori Mitsuda just poured his heart on and wrote it down on paper.
           
__label__2 an absolute masterpiece: I am quite sure any of you actually
           taking the time to read this have played the game at least once,
           and heard at least a few of the tracks here.
           And whether you were aware of it or not, Mitsuda's music
           contributed greatly to the mood of every single minute
           of the whole game.Composed of 3 CDs and quite a few songs
           (I haven't an exact count), all of which are heart-rendering
           and impressively remarkable, this soundtrack is one I assure
           you you will not forget. It has everything for every listener
           -- from fast-paced and energetic (Dancing the Tokage or Termina Home),
           to slower and more haunting (Dragon God), to purely beautifully
           composed (Time's Scar), to even some fantastic vocals
           (Radical Dreamers).This is one of the best videogame
           soundtracks out there, and surely Mitsuda's best ever. ^_^
           
__label__1 Buyer beware: This is a self-published book,
           and if you want to know why--read a few paragraphs!
           Those 5 star reviews must have been written by Ms. Haddon's
           family and friends--or perhaps, by herself! I can't imagine
           anyone reading the whole thing--I spent an evening with
           the book and a friend and we were in hysterics reading bits
           and pieces of it to one another. It is most definitely bad
           enough to be entered into some kind of a "worst book" contest.
           I can't believe Amazon even sells this kind of thing.
           Maybe I can offer them my 8th grade term paper on
           "To Kill a Mockingbird"--a book I am quite sure Ms. Haddon
           never heard of. Anyway, unless you are in a mood to send a book
           to someone as a joke---stay far, far away from this one!
```

</details>

