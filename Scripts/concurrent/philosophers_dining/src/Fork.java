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

			if (used == true) {
				wait();
			} else {
				used = true;
				this.currentPhilosophe = philosophe;
				this.io.print("The fork number " + this.index + " is taken by the philosophe "
						+ this.currentPhilosophe.getIndex());
				notifyAll();
			}

		} catch (Exception e) {
		}
	}

	public synchronized void leaveFork() {
		if (used == true) {
			used = false;
			this.currentPhilosophe = null;
			this.io.print("The fork number " + this.index + " has been released");
			notifyAll();
		}
	}
}
