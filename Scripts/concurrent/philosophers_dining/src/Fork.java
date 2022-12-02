public class Fork {
	private boolean used = false;
	private int index;
	private IO io;
	private Philosophe currentPhilosophe;

	public Fork(int index, IO io) {
		this.index = index;
		this.io = io;
	}

	public synchronized void useFork(Philosophe philosophe) {
		try {
			System.out.println("The philosopher " + philosophe.getIndex() + " is trying to use the fork " + this.index);
			while (used) {
				try {
					System.out.println("The philosopher " + philosophe.getIndex() + " is waiting to use the fork " + this.index);
					wait();
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					System.err.println("Thread Interrupted");
				}
			}
			used = true;
			this.currentPhilosophe = philosophe;
			this.io.print(
					"The philosopher " + this.currentPhilosophe.getIndex() + " took the fork number " + this.index);
			// Notify all threads that use this instance, not this object.
			notifyAll();
		} catch (Exception e) {
		}
	}

	public synchronized void leaveFork(Philosophe philosophe) {
		if (used == true) {
			used = false;
			this.currentPhilosophe = null;
			System.out.println("The philosopher " + philosophe.getIndex() + " released the fork number " + this.index);
			notifyAll();
		}
	}
}
