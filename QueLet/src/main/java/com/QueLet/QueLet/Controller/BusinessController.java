package com.QueLet.QueLet.Controller;

import com.QueLet.QueLet.Model.Business;
import com.QueLet.QueLet.Service.BusinessService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
@RestController
@RequestMapping("/business")
public class BusinessController {
    @Autowired
    private BusinessService businessService;

    @GetMapping("/All-business")
    public List<Business> getAllBusiness(){
        return businessService.AllBusiness();
    }
}
