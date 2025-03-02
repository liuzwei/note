CREATE TABLE `t_area` (
  `id` int NOT NULL AUTO_INCREMENT,
  `area_name` varchar(100) DEFAULT NULL COMMENT '名称',
  `area_code` varchar(100) DEFAULT NULL COMMENT '编码',
  `parent_code` varchar(100) DEFAULT NULL COMMENT '父级编码',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO `area`.`t_area` (`id`, `area_name`, `area_code`, `parent_code`) VALUES (1, '北京', '11', '0');
INSERT INTO `area`.`t_area` (`id`, `area_name`, `area_code`, `parent_code`) VALUES (2, '北京市', '1101', '11');
INSERT INTO `area`.`t_area` (`id`, `area_name`, `area_code`, `parent_code`) VALUES (3, '东城区', '110101', '1101');
INSERT INTO `area`.`t_area` (`id`, `area_name`, `area_code`, `parent_code`) VALUES (4, '西城区', '110102', '1101');
INSERT INTO `area`.`t_area` (`id`, `area_name`, `area_code`, `parent_code`) VALUES (5, '朝阳区', '110103', '1101');



CREATE TABLE `t_user` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) DEFAULT NULL,
  `age` varchar(100) DEFAULT NULL,
  `from` varchar(100) DEFAULT NULL,
  `area_code` varchar(255) DEFAULT NULL COMMENT '行政区划',
  `user_status` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;