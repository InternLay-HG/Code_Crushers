package com.QueLet.QueLet.Repository;

import com.QueLet.QueLet.Model.Business;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface BusinessRepository extends JpaRepository<Business, Long>{
    Optional<Business> findByUsername(String username);
}