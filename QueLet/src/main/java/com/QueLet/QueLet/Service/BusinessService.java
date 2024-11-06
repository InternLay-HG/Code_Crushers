package com.QueLet.QueLet.Service;

import com.QueLet.QueLet.Model.Business;
import com.QueLet.QueLet.Model.Customer;
import com.QueLet.QueLet.Repository.BusinessRepository;
import com.QueLet.QueLet.Repository.CustomerRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
@Service
public class BusinessService {
    @Autowired
    private BusinessRepository businessRepository;
    @Autowired
    private PasswordEncoder passwordEncoder;


    public Business save(Business user) {
        if (user.getPassword() == null || user.getPassword().isEmpty()) {
            throw new IllegalArgumentException("password cannot be null or empty");
        }
        Optional<Business> existingUser = businessRepository.findByUsername(user.getUsername());

        if (existingUser.isPresent()) {
            throw new IllegalArgumentException("username already taken");
        }
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        return businessRepository.save(user);
    }

    public List<Business> AllBusiness() {
        return businessRepository.findAll();
    }
}


