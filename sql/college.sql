-- database: :memory:
-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Apr 14, 2024 at 12:23 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `college`
--

-- --------------------------------------------------------

--
-- Table structure for table `attendance`
--

CREATE TABLE `attendance` (
  `id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `start_time` varchar(20) DEFAULT NULL,
  `end_time` varchar(20) DEFAULT NULL,
  `date` date DEFAULT curdate(),
  `roll_no` varchar(20) NOT NULL,
  `division` varchar(10) DEFAULT NULL,
  `branch` varchar(100) DEFAULT NULL,
  `reg_id` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `attendance`
--

INSERT INTO `attendance` (`id`, `name`, `start_time`, `end_time`, `date`, `roll_no`, `division`, `branch`, `reg_id`) VALUES
(1, 'Govind Choudhary ', '10:12:35', NULL, '2024-04-02', '6', 'TE2', 'ECS', '16199'),
(2, 'Parth Ghadi', '10:12:36', NULL, '2024-04-02', '12', 'TE2', 'ECS', '16122'),
(3, 'Rushikesh Gaikwad', '10:31:15', NULL, '2024-04-02', '11', 'TE2', 'ECS', '16489'),
(4, 'Parth ', '10:31:18', NULL, '2024-04-02', '12', 'TE2', 'ECS', '16122'),
(5, 'Parth Ghadi', '10:15:10', NULL, '2024-04-05', '12', 'TE2', 'ECS', '16122'),
(6, 'Prathmesh Gilbile', '10:17:29', NULL, '2024-04-05', '13', 'TE2', 'ECS', '16907'),
(7, 'Rushikesh Gaikwad', '10:23:56', NULL, '2024-04-05', '11', 'TE2', 'ECS', '16489'),
(8, 'Govind Choudhary ', '10:24:02', NULL, '2024-04-05', '6', 'TE2', 'ECS', '16199'),
(9, 'Parth Ghadi', '10:24:42', NULL, '2024-04-06', '12', 'TE2', 'ECS', '16122'),
(10, 'Parth Shinde', '11:43:00', NULL, '2024-04-06', '52', 'TE2', 'ECS', '16415'),
(11, 'Prasanna Tribhuvan', '11:58:26', NULL, '2024-04-06', '59', 'TE2', 'ECS', '16380'),
(12, 'Anuraag Patil', '11:58:32', NULL, '2024-04-06', '34', 'TE2', 'ECS', '16658');

-- --------------------------------------------------------

--
-- Table structure for table `student_data`
--

CREATE TABLE `student_data` (
  `id` int(10) NOT NULL,
  `name` varchar(25) NOT NULL,
  `rollno` varchar(5) NOT NULL,
  `division` varchar(10) NOT NULL,
  `branch` varchar(15) NOT NULL,
  `regid` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `student_data`
--

INSERT INTO `student_data` (`id`, `name`, `rollno`, `division`, `branch`, `regid`) VALUES
(1, 'Parth Ghadi', '12', 'TE2', 'ECS', '16122'),
(2, 'Ishica ', '15', 'TE2', 'ECS', '16210'),
(3, 'Praik Niveragi', '32', 'TE2', 'ECS', '16400'),
(4, 'Shruti Patil', '35', 'TE2', 'ECS', '16464'),
(5, 'Anuraag Patil', '34', 'TE2', 'ECS', '16658'),
(6, 'Mantasha Jariwala', '18', 'TE2', 'ECS', '16811'),
(7, 'Saloni Dawale', '7', 'TE2', 'ECS', '16894'),
(8, 'Prathmesh Gilbile', '13', 'TE2', 'ECS', '16907'),
(9, 'Chaitanya Koli', '23', 'TE2', 'ECS', '16802'),
(10, 'Karishma Rathod', '37', 'TE2', 'ECS', '17704'),
(11, 'Ritika ', '40', 'TE2', 'ECS', '16462'),
(12, 'Vishal Talekar', '58', 'TE2', 'ECS', '16574'),
(13, 'Parth Shinde', '52', 'TE2', 'ECS', '16415'),
(14, 'Prasanna Tribhuvan', '59', 'TE2', 'ECS', '16380');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(100) DEFAULT NULL,
  `reg_id` int(11) DEFAULT NULL,
  `psw` varchar(128) DEFAULT NULL,
  `role` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `reg_id`, `psw`, `role`) VALUES
(4, 'Unknown', 10000, '123', 'student');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `attendance`
--
ALTER TABLE `attendance`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `student_data`
--
ALTER TABLE `student_data`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `reg_id` (`reg_id`),
  ADD UNIQUE KEY `role` (`role`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `attendance`
--
ALTER TABLE `attendance`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;

--
-- AUTO_INCREMENT for table `student_data`
--
ALTER TABLE `student_data`
  MODIFY `id` int(10) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
