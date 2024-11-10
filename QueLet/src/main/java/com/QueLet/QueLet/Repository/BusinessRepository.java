package com.QueLet.QueLet.Repository;

import com.QueLet.QueLet.Model.Business;
import com.QueLet.QueLet.Model.CategoryType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;
import java.util.Optional;

public interface BusinessRepository extends JpaRepository<Business, Long>{
    Optional<Business> findByUsername(String username);
    List<Business> findByTypeOfBusiness(CategoryType categoryType);
}