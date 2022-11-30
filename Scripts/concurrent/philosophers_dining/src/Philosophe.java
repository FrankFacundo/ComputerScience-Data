import java.util.Random;

public class Philosophe extends Thread{

	private Fork forkRight;
	private Fork forkLeft;
	private int index;
	private IO io;
	private Random rand = new Random();
	private int timeRandomMax = 1000;
	private int numberOfMeals = 0;
	
	public Philosophe (int index, Fork forkRight, Fork forkLeft, IO io)
	{
		this.index = index;
		this.forkRight = forkRight;
		this.forkLeft = forkLeft;
		this.io = io;
	}
	
	public void takeForkRight()
	{
		this.forkRight.useFork(this);
	}
	
	public void takeForkLeft()
	{
		this.forkLeft.useFork(this);
	}
	
	public void think()
	{
		this.io.print("The philosopher " + this.index + " is thinking");
		try {
			Thread.sleep(rand.nextInt(timeRandomMax));
		} catch (Exception e) {}
	}
	
	public void eat() 
	{
		this.io.print("The philosopher " + this.index + " is eating");
		try {
			Thread.sleep(rand.nextInt(timeRandomMax));
		} catch (Exception e) {}
		this.numberOfMeals = this.numberOfMeals + 1; 
	}
	
	public void leaveFork()
	{
		this.forkRight.leaveFork();
		this.forkLeft.leaveFork();
		this.io.print("The philosopher " + this.index + " has released both forks.");
	}
	
	public void writeMeals()
	{
		this.io.print("The philosopher " + this.index + " has eaten " + Integer.toString(numberOfMeals) + " meals.");
	}
	
	public int getIndex()
	{
		return this.index;
	}
	
	public void run()
	{
		long startTime = System.nanoTime();
		long endTime = System.nanoTime();

		while(endTime - startTime < 1000000000*10)
		{
			this.think();
			this.takeForkLeft();
			this.takeForkRight();
			this.eat();
			this.leaveFork();
			endTime = System.nanoTime();
		}
		
		try {
			Thread.sleep(2000);
		} catch (InterruptedException e) {e.printStackTrace();}
		
		this.writeMeals();
		
	}
}
