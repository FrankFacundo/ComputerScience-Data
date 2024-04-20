CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,    -- A unique identifier for each transaction
    client_id INT NOT NULL,               -- An integer ID for the client
    client_name VARCHAR(255) NOT NULL,    -- The client's name, with a maximum length of 255 characters
    amount DECIMAL(10, 2) NOT NULL,       -- The transaction amount with two decimal places
    beneficiary VARCHAR(255) NOT NULL,    -- The beneficiary's name of the transaction
    country VARCHAR(100) NOT NULL,        -- The country of the transaction
    transaction_time TIMESTAMP NOT NULL   -- Timestamp of the transaction
);

-- Seed data into the transactions table
INSERT INTO transactions (client_id, client_name, amount, beneficiary, country, transaction_time) VALUES
(101, 'John Doe', 1200.00, 'My Contractor', 'USA', '2023-01-15 14:30:00'),
(101, 'John Doe', -200.00, 'Jane Smith', 'USA', '2023-01-16 09:20:00'),
(101, 'John Doe', -800.00, 'Proprietaire', 'USA', '2023-01-17 12:00:00'),
(101, 'John Doe', -100.00, 'Achats', 'USA', '2023-01-18 15:45:00'),
(102, 'Alice Johnson', 750.00, 'Nancy Drew', 'Canada', '2023-01-19 11:00:00'),
(103, 'Xiao Ming', 560.50, 'Chen Wei', 'China', '2023-01-20 16:30:00'),
(104, 'Anil Gupta', 310.00, 'Raj Patel', 'India', '2023-01-21 14:00:00');
