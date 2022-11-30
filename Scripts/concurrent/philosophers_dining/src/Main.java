public class Main {

	public static void main(String[] args) {
		// Number of philosophers
		int numberIndex = 5;
		// Create table of fork and philosophers
		Fork[] forks = new Fork[numberIndex];
		Philosophe[] philosophers = new Philosophe[numberIndex];

		// Create objects
		// "false" to print in console and "true" to print in file
		IO io = new IO(false);

		for (int i = 0; i < numberIndex; i++) {
			forks[i] = new Fork(i, io);
		}

		for (int i = 0; i < numberIndex; i++) {
			philosophers[i] = new Philosophe(i, forks[i], forks[(i + 1) % numberIndex], io);
		}

		io.writeHeading();

		for (int i = 0; i < numberIndex; i++) {
			philosophers[i].start();
		}
	}
}
